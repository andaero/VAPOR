import numpy
import tensorflow as tf
from VAPORModel import VAPOR_Model
import numpy as np
from Supply_VAPOR_Model_Preprocess import model_preprocess_CNN, model_preprocess, model_preprocess_CNN_Twenty

input_len = 96

model = VAPOR_Model(SLAM_Dense=32, ConvBlock_Size=128, filterSize=2,
                    CNN_Dense=32, CNN_2=128, tensorLen=int(input_len / 4), Conv2Bool=False)
model.load_weights("SupplyVAPOR/Liquid-624.83-RMSE")

train_x_pv, validation_x_pv, train_x_aux, validation_x_aux, train_y, validation_y = model_preprocess_CNN(input_len, supplyTotal=True, showFig=False, normalize=True)


val_pv = tf.convert_to_tensor(numpy.expand_dims(validation_x_pv[0], axis=0))
val_aux = tf.convert_to_tensor(numpy.expand_dims(validation_x_aux[0], axis=0))

print(val_pv.shape)
print(val_aux.shape)

print("BEFORE PREDICTION")

prediction = model.predict([val_pv, val_aux])
print(prediction)

