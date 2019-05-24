# coding=utf-8

import tensorflow as tf
import numpy as np
import trading_env
import pandas as pd
from datetime import datetime
import os

from algorithm import config
# from base.env.market import Market
from checkpoints import CHECKPOINTS_DIR
from base.algorithm.model import BaseRLTFModel
from helper.args_parser import model_launcher_parser
from helper.data_logger import generate_algorithm_logger, generate_market_logger


class Algorithm(BaseRLTFModel):

    def __init__(self, session, env, a_space, s_space, **options):
        super(Algorithm, self).__init__(session, env, a_space, s_space, **options)

        self.actor_loss, self.critic_loss = .0, .0

        # Initialize buffer.
        self.buffer = np.zeros((self.buffer_size, self.s_space * 2 + 1 + 1))
        self.buffer_length = 0
        # self.alpha = []

        self._init_input()
        self._init_nn()
        self._init_op()
        self._init_saver()
        self._init_summary_writer()

    def _init_input(self):
        self.s = tf.placeholder(tf.float32, [None, self.s_space], 'state')
        self.r = tf.placeholder(tf.float32, [None, 1], 'reward')
        self.s_next = tf.placeholder(tf.float32, [None, self.s_space], 'state_next')

    def _init_nn(self):
        # Initialize predict actor and critic.
        self.a_predict = self.__build_actor_nn(self.s, "predict/actor", trainable=True)
        self.q_predict = self.__build_critic(self.s, self.a_predict, "predict/critic", trainable=True)
        # Initialize target actor and critic.
        self.a_next = self.__build_actor_nn(self.s_next, "target/actor", trainable=False)
        self.q_next = self.__build_critic(self.s_next, self.a_next, "target/critic", trainable=False)
        # Save scopes
        self.scopes = ["predict/actor", "target/actor", "predict/critic", "target/critic"]

    def _init_op(self):
        # Get actor and critic parameters.
        params = [tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope) for scope in self.scopes]
        zipped_a_params, zipped_c_params = zip(params[0], params[1]), zip(params[2], params[3])
        # Initialize update actor and critic op.
        self.update_a = [tf.assign(t_a, (1 - self.tau) * t_a + self.tau * p_a) for p_a, t_a in zipped_a_params]
        self.update_c = [tf.assign(t_c, (1 - self.tau) * t_c + self.tau * p_c) for p_c, t_c in zipped_c_params]
        # Initialize actor loss and train op.
        with tf.variable_scope('actor_loss'):
            self.a_loss = -tf.reduce_mean(self.q_predict)
        with tf.variable_scope('actor_train'):
            self.a_train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.a_loss, var_list=params[0])
        # Initialize critic loss and train op.
        self.q_target = self.r + self.gamma * self.q_next
        with tf.variable_scope('critic_loss'):
            self.c_loss = tf.losses.mean_squared_error(self.q_target, self.q_predict)
        with tf.variable_scope('critic_train'):
            self.c_train_op = tf.train.RMSPropOptimizer(self.learning_rate * 2).minimize(self.c_loss, var_list=params[2])
        # Initialize variables.
        self.session.run(tf.global_variables_initializer())

    def run(self):
        time = self.env.df['datetime'].dt.date
        if self.mode != 'train':
            self.restore()
            self.env.reset()
            self.env.render()
            s, r, done, info = self.env.step(0)
            # s = self.env.df.iloc[s[0][3]][1:6]
            # s = [np.array(s)]  # let(5,) -> (?,5)
            while True:
                if (time[s[0][3]] != time[s[0][3] + 2]):
                    if s[0][6] < 0:
                        s, r, done, info = self.env.step(1)
                    elif s[0][6] > 0:
                        s, r, done, info = self.env.step(2)
                    else:
                        s, r, done, info = self.env.step(0)
                else:
                    state = self.env.df.iloc[s[0][3]][1:6]
                    state = [np.array(state)]  # let(5,) -> (?,5)
                    #~ c, a, a_index = self.predict(state)
                    # a = self.predict(state)
                    # a_index = a
                    # s_next, r, status, info = self.env.forward(c, a)]
                    #~ s, r, done, info = self.env.step(a)
                    a = self.predict(state)
                    a_index = a
                    # a = pd.Series(a)
                    self.env.alpha.append(a)
                    df = pd.DataFrame(self.env.alpha)
                    c = df.rolling(window=3).mean()
                    d = df.rolling(window=7).mean()
                    # self.env.alpha.join(a)
                    # s, r, done, info = self.env.step(0)
                    # print(c.lioc[-1])
                    if c.count()[0] > 10:
                        if (c.iloc[-1][0] > d.iloc[-1][0]) & (c.iloc[-2][0] < d.iloc[-2][0]):
                            s, r, done, info = self.env.step(1)
                        elif (c.iloc[-1][0] < d.iloc[-1][0]) & (c.iloc[-2][0] > d.iloc[-2][0]):
                            s, r, done, info = self.env.step(2)
                        else:
                            s, r, done, info = self.env.step(0)
                    else:
                        s, r, done, info = self.env.step(0)
                self.env.render()

                if done == 1:
                    # self.env.trader.log_asset(episode)
                    break
        else:
            for episode in range(self.episodes):
                self.log_loss(episode)
                self.env.reset()
                reward =[]
                # if episode == 40:
                #     self.env.render()
                # self.env.render()
                s, r, done, info = self.env.step(0)
                # idx = s[0][3]
                # s = self.env.df.iloc[s[0][3]][1:6]
                # np.expand_dims(s, axis=1)
                # s = np.reshape(s,(1,5))
                # s = s[: , np.newaxis]
                # s = [np.array(s)]#let(5,) -> (?,5)
                # s.reshape((1,1,5))
                while True:
                    if(time[s[0][3]] != time[s[0][3]+2]):
                        if s[0][6] < 0:
                            s, r, done, info = self.env.step(1)
                        elif s[0][6] > 0:
                            s, r, done, info = self.env.step(2)
                        else :
                            s, r, done, info = self.env.step(0)
                    else:
                        state = self.env.df.iloc[s[0][3]][1:6]
                        state = [np.array(state)]  # let(5,) -> (?,5)
                        # c, a, a_index = self.predict(state) #
                        a = self.predict(state)
                        a_index = a
                        # a = pd.Series(a)
                        self.env.alpha.append(a)
                        df = pd.DataFrame(self.env.alpha)
                        c = df.rolling(window=3).mean()
                        d = df.rolling(window=7).mean()
                        # self.env.alpha.join(a)
                        # s, r, done, info = self.env.step(0)
                        # print(c.lioc[-1])
                        if c.count()[0] > 10:
                            if (c.iloc[-1][0] > d.iloc[-1][0]) & (c.iloc[-2][0] < d.iloc[-2][0]):
                                s, r, done, info = self.env.step(1)
                            elif(c.iloc[-1][0] < d.iloc[-1][0]) & (c.iloc[-2][0] > d.iloc[-2][0]):
                                s, r, done, info = self.env.step(2)
                            else:
                                s, r, done, info = self.env.step(0)
                        else:
                            s, r, done, info = self.env.step(0)

                        # cross a above some thing do a
                        #
                        #####################################
                        # s_next, r, status, info = self.env.forward(c, a)]
                        # s, r, done, info = self.env.step(a)

                    # if episode == 1:
                    #     self.env.render()

                    # item = lambda r : r if r != 0 else n
                    r = r + s[0][-3]
                    # reward.append(r)
                    # r = np.cumsum(reward)
                    # if r != 0:
                    #     reward.append(r)
                        # r_cum = np.cumsum(reward)
                        # if r == 0:
                        #     r_cum[-1]+= s[0][-3]
                    # r = np.sqrt(1) * reward.mean() / reward.std()
                    # if len(reward) > 0:
                    #     r = np.sqrt(len(reward)) * np.mean(reward) / np.std(reward)
                        # r = r*100
                        # print(r)
                    if np.isnan(r):
                        r = 0
                    # print(r)
                    # self.env.render()
                    state = [self.env.df.iloc[s[0][3]-1][1:6]]
                    state_next = [self.env.df.iloc[s[0][3]][1:6]]
                    # self.save_transition(s, a_index, r, s_next)
                    # print(a_index)
                    self.save_transition(state, a_index, r, state_next)
                    self.train()
                    # state = state_next
                    if done == 1 :
                        #self.env.trader.log_asset(episode)
                        break
                if self.enable_saver and episode % 10 == 0:
                    self.save(episode)

    def train(self):
        if self.buffer_length < self.buffer_size:
            return
        self.session.run([self.update_a, self.update_c])
        s, a, r, s_next = self.get_transition_batch()
        self.critic_loss, _ = self.session.run([self.c_loss, self.c_train_op], {self.s: s, self.a_predict: a, self.r: r, self.s_next: s_next})
        self.actor_loss, _ = self.session.run([self.a_loss, self.a_train_op], {self.s: s})

    def predict(self, s):
        a = self.session.run(self.a_predict, {self.s: s})[0][0]
        # return self.get_stock_code_and_action(a, use_greedy=True, use_prob=True if self.mode == 'train' else False)
        return a

    def save_transition(self, s, a, r, s_next):
        transition = np.hstack((s, [[a]], [[r]], s_next))
        self.buffer[self.buffer_length % self.buffer_size, :] = transition
        self.buffer_length += 1

    def get_transition_batch(self):
        # indices = np.random.choice(self.buffer_size, size=self.batch_size)
        indices = np.random.choice(self.buffer_size-self.batch_size, size=1)
        indices = np.arange(indices,indices+self.batch_size)
        batch = self.buffer[indices, :]
        s = batch[:, :self.s_space]
        a = batch[:, self.s_space: self.s_space + 1]
        r = batch[:, -self.s_space - 1: -self.s_space]
        s_next = batch[:, -self.s_space:]
        return s, a, r, s_next

    def log_loss(self, episode):
        self.logger.warning("Episode: {0} | Actor Loss: {1:.2f} | Critic Loss: {2:.2f}".format(episode,
                                                                                               self.actor_loss,
                                                                                               self.critic_loss))

    def __build_actor_nn(self, state, scope, trainable=True):

        w_init, b_init = tf.random_normal_initializer(.0, .001), tf.constant_initializer(.1)

        with tf.variable_scope(scope):
            # state is ? * code_count * data_dim.
            first_dense = tf.layers.dense(state,
                                          64,
                                          tf.nn.relu,
                                          kernel_initializer=w_init,
                                          bias_initializer=b_init,
                                          trainable=trainable)

            action = tf.layers.dense(first_dense,
                                     1,
                                     None,#tf.nn.sigmoid,
                                     kernel_initializer=w_init,
                                     bias_initializer=b_init,
                                     trainable=trainable)

            # return tf.multiply(action, self.a_space - 1)
            return tf.multiply(action, 1)

    @staticmethod
    def __build_critic(state, action, scope, trainable=True):

        w_init, b_init = tf.random_normal_initializer(.0, .3), tf.constant_initializer(.1)

        with tf.variable_scope(scope):

            s_first_dense = tf.layers.dense(state,
                                            32,
                                            tf.nn.relu,
                                            kernel_initializer=w_init,
                                            bias_initializer=b_init,
                                            trainable=trainable)

            a_first_dense = tf.layers.dense(action,
                                            32,
                                            tf.nn.relu,
                                            kernel_initializer=w_init,
                                            bias_initializer=b_init,
                                            trainable=trainable)

            q_value = tf.layers.dense(tf.nn.relu(s_first_dense + a_first_dense),
                                      1,
                                      kernel_initializer=w_init,
                                      bias_initializer=b_init,
                                      trainable=trainable)

            return q_value


def main():
    # mode = args.mode
    mode = 'test'#'train'
    # codes = args.codes
    codes = ["AU88", "RB88", "CU88", "AL88"]
    # codes = ["T9999"]
    # market = args.market
    market = 'future'
    # episode = args.episode
    episode = 2000
    training_data_ratio = 0.5
    # training_data_ratio = args.training_data_ratio

    model_name = os.path.basename(__file__).split('.')[0]

    df = pd.read_csv('D:\[AIA]\workspace\TradingGym\dataset\TXF1_5m.csv', parse_dates=[['Date', 'Time']])
    df = df.iloc[-10001:]
    df.reset_index(drop=True, inplace=True)
    df = df.rename(columns={'Date_Time': 'datetime'})
    df['serial_number'] = df.index
    # df['serial_number'] = df['serial_number'] - df['serial_number'].iloc[0]
    env = trading_env.make(env_id='training_v1', obs_data_len=1, step_len=1,
                           df=df, fee=2, max_position=1, deal_col_name='Price',
                           feature_names=['Price', 'Volume',  # 'datetime' ,
                                          'Open', 'serial_number',
                                          'High', 'Low'],
                           fluc_div=100.0)
    # env = Market(codes, start_date="2012-01-01", end_date="2018-01-01", **{
    #     "market": market,
    #     # "use_sequence": True,
    #     "logger": generate_market_logger(model_name),
    #     "training_data_ratio": training_data_ratio,
    # })

    algorithm = Algorithm(tf.Session(config=config), env, 1, 5, **{
        "mode": mode,
        "episodes": episode,
        "enable_saver": True,
        "learning_rate": 0.003,
        "batch_size":1320,
        "enable_summary_writer": True,
        "logger": generate_algorithm_logger(model_name),
        "save_path": os.path.join(CHECKPOINTS_DIR, "RL", model_name, market, "model"),
        "summary_path": os.path.join(CHECKPOINTS_DIR, "RL", model_name, market, "summary"),
    })

    algorithm.run()
    algorithm.eval()
    # algorithm.plot()


if __name__ == '__main__':
    # main(model_launcher_parser.parse_args())
    main()

