import mongoengine

# mongoengine.connect(db="quant", host="0.0.0.0", username='', password='')
mongoengine.connect(db="quant", host="mongodb://127.0.0.1:27017/")