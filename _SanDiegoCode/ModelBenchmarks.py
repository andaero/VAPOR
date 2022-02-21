import numpy
import math

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model

from VAPORModel import VAPOR_Model
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from Supply_VAPOR_Model_Preprocess import model_preprocess_CNN, model_preprocess, model_preprocess_CNN_Twenty


input_len = 168

model = VAPOR_Model(SLAM_Dense=32, ConvBlock_Size=128, filterSize=2,
                    CNN_Dense=64, CNN_2=128, tensorLen=int(input_len/4), Conv2Bool=False)

# model.load_weights("SupplyVAPOR/LSTM-709.75-RMSE")
# model.load_weights("SupplyVAPOR/ANN-748.27-RMSE")

model.load_weights("SupplyVAPOR/Liquid-548.52-RMSE")

train_x_pv, validation_x_pv, train_x_aux, validation_x_aux, train_y, validation_y = model_preprocess_CNN(input_len, supplyTotal=True, showFig=False, normalize=True)





print(validation_x_pv.shape)
print(validation_x_aux.shape)

print("BEFORE PREDICTION")
print("SIZE ", validation_x_pv.shape[0])
prediction = []
count = 0
maxError = 0
maxPred = 0
maxReal = 0
diff = 0
total = 0
for i in range (validation_x_pv.shape[0]):
    print(i)

    val_pv = numpy.expand_dims(validation_x_pv[i], axis=0)
    val_aux = numpy.expand_dims(validation_x_aux[i], axis=0)
    pred = model.predict([val_pv, val_aux])[0][0]

    if(pred<0):
        pred = 0
    diff += abs(pred-validation_y[i])
    total += validation_y[i]
    # if(abs(pred-validation_y[i])> maxError):
    #     maxError = abs(pred-validation_y[i])
    #     maxPred = pred
    #     maxReal = validation_y[i]
    prediction.append(pred)
# print("Max error", maxError)
# print("Max pred", maxPred)
# print("Max real ", maxReal)
print("RMSE:", mean_squared_error(validation_y, prediction, squared=False))
print("Coefficient of Determination (r^2): ", r2_score(validation_y, prediction))
print("Total diff: ", diff)
print("Total energy: ", total)

