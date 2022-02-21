import numpy
import math

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model

from VAPORModel import VAPOR_Model
from keras.models import Model
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import keract

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
for i in range (1):
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

    from keras import backend as K

    inp = model.input  # input placeholder
    outputs = [layer.output for layer in model.layers]  # all layer outputs
    functor = K.function([inp, K.learning_phase()], outputs)  # evaluation function

    # Testing
    test = [val_pv, val_aux]
    layer_outs = functor([test, 1.])
    print (layer_outs)

    prediction.append(pred)