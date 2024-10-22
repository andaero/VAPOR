import numpy
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model

from VAPORModel import VAPOR_Model
import numpy as np
from numpy import savetxt

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
# for i in range(150):

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

SMALL_SIZE = 10
MEDIUM_SIZE = 13
LARGE_SIZE = 15
def PredictionVsReal():
    plt.figure(figsize=(8,5))
    plt.rcParams["font.family"] = "Sans"

    plt.rc('font', size=SMALL_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels

    plt.xlabel("Time(hours)", fontsize=MEDIUM_SIZE)
    plt.ylabel("Energy Generation(kwh)", fontsize=MEDIUM_SIZE)

    #plt.legend()



    #Prediction line
    plt.plot(prediction[90:150], color="darkturquoise", label="Prediction")

    # Real line
    plt.plot(validation_y[90:150], color="darkorange", label="Real")

    plt.legend()

    plt.title("Comparison Between Predicted and Real Energy Generation", fontsize=LARGE_SIZE)
    # plt.savefig("Graphs/SupplyGraphPredvReal.png")

    plt.show()

PredictionVsReal()

npPred = np.array(prediction)
print(npPred)
npVal = np.array(validation_y)
savetxt('pred.csv', npPred, fmt='%f')
savetxt('real.csv', npVal, fmt='%f')

# print("Max error", maxError)
# print("Max pred", maxPred)
# print("Max real ", maxReal)
# print("RMSE:", mean_squared_error(validation_y, prediction, squared=False))
# print("Coefficient of Determination (r^2): ", r2_score(validation_y, prediction))
# print("Total diff: ", diff)
# print("Total energy: ", total)

