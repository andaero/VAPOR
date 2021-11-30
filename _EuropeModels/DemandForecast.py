from collections import deque
import glob
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, ConvLSTM2D, MaxPool2D, MaxPool3D, Flatten, Bidirectional
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras import mixed_precision
from functools import reduce

from scipy import stats

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

drop_remainder = True

trainDf = pd.read_csv("../EuropeData/train.csv", parse_dates=["time"], usecols=["time", "consumption", "temp", "precip_1h:mm"])
print(trainDf)

trainDf['Weekday'] = trainDf["time"].dt.weekday #.map(weekdays) to show which physical day it is
trainDf['day'] = trainDf["time"].dt.day
trainDf['hour'] = trainDf["time"].dt.hour

trainDf = trainDf.drop(["time"],axis=1)


#Create a target column for load in future
future = 24 #predicting 24 hours in the future
seq_len = 48 #take the last 24 hours of info

trainDf["target"] = trainDf["consumption"].shift(-future)
trainDf.dropna(inplace=True)

print(trainDf.head())



#Figuring out which parts of data to use for prediction vs results
print("Moving on to sorting data")
# Scale the nums to be between 0-1
scaler = MinMaxScaler()
trainDf["consumption"] = scaler.fit_transform(trainDf["consumption"].values.reshape(-1,1))
trainDf["temp"] = scaler.fit_transform(trainDf["temp"].values.reshape(-1,1))
trainDf["precip_1h:mm"] = scaler.fit_transform(trainDf["precip_1h:mm"].values.reshape(-1,1))


#df["target"] = scaler.fit_transform(df["Temp"].values.reshape(-1,1))

# df["Weekday"] = scaler.fit_transform(df["Weekday"].values.reshape(-1,1))
# df["Month"] = scaler.fit_transform(df["Weekday"].values.reshape(-1,1))
# df["Day"] = scaler.fit_transform(df["Weekday"].values.reshape(-1,1))
# df["Hour"] = scaler.fit_transform(df["Weekday"].values.reshape(-1,1))


# df.plot(y=["Load", "Temp"])
# plt.show()

times = trainDf.index.values
last_10 = trainDf.index.values[-int(0.1*len(times))]
validation_df = trainDf[(trainDf.index >= last_10)]
main_df = trainDf[(trainDf.index< last_10)]

print(validation_df)
print(main_df)

def preprocess(df):

    sequential_data = []  # this is a list that will CONTAIN the sequences
    prev_days = deque( maxlen=seq_len)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in

    for i in df.to_numpy():  # iterate over the values
        prev_days.append([n for n in i[:-1]])  # store all but the target
        if len(prev_days) == seq_len:  # make sure we have 24 sequences!
            sequential_data.append([np.array(prev_days), i[-1]])  # append those bad boys!

    random.shuffle(sequential_data)  # shuffle for good measure.
    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), y

train_x, train_y = preprocess(main_df)
validation_x, validation_y = preprocess(validation_df)

print(f"train data: {len(train_x)}, validation: {len(validation_x)}")

train_x = np.asarray(train_x)
train_y = np.asarray(train_y)
validation_x = np.asarray(validation_x)
validation_y = np.asarray(validation_y)

print(train_x)
print(train_y)
print(train_x.shape, train_y.shape, validation_x.shape, validation_y.shape)

#CREATING THE MODEL

class SaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self, save_best_metric='val_mean_absolute_percentage_error'):
        self.save_best_metric = save_best_metric
        self.lowestLoss = 1.2
    def on_epoch_end(self, epoch, logs=None):
        #print(logs[self.save_best_metric])
        if(logs[self.save_best_metric] < self.lowestLoss):
            self.lowestLoss = logs[self.save_best_metric]
            self.model.save(f"models/model{round(logs[self.save_best_metric],2)}.model")
            print(f"Model with MAPE of {logs[self.save_best_metric]} saved")

save_best_model = SaveBestModel()

EPOCHS = 200
BatchSizes = [32]
learning_rs = [0.00001]
layers = [2,3]
dense = 2;
dBatchSize = 1;



for BatchSize in BatchSizes:
    for learning_r in learning_rs:
        for layer in layers:
            model = Sequential()
            NAME = f"ANN-Dense Layers-{dense}-Dense Batch-{dBatchSize}-SeqLen-{seq_len}Filters-{BatchSize}-Layers{layer}-Learning Rate-{learning_r}-Time-{int(time.time())}"
            model.add(LSTM(BatchSize, input_shape=(train_x.shape[1:]), return_sequences=True))
            model.add(Dropout(0.1))
            model.add(BatchNormalization())
            for x in range(layer-1):
                model.add(Bidirectional(LSTM(BatchSize, input_shape=(train_x.shape[1:]), return_sequences=True)))
            model.add(LSTM(BatchSize, input_shape=(train_x.shape[1:]), return_sequences=True))
            model.add(Dropout(0.1))
            model.add(BatchNormalization())

            #PURE BASIC ANN - 3.7% MAPE
            #model.add(tf.keras.Input(shape=train_x.shape[1:]))
            model.add(Flatten())
            model.add(Dense(BatchSize, activation="linear"))


            model.add(Dense(dBatchSize, activation="linear"))



            opt = tf.keras.optimizers.Adam(learning_rate= learning_r, decay=1e-6)
            model.compile(loss="mse", optimizer=opt, metrics=[tf.keras.metrics.RootMeanSquaredError(),tf.keras.metrics.MeanAbsolutePercentageError()]) #tf.keras.metrics.RootMeanSquaredError()

            model.summary()

            tensorboard = TensorBoard(log_dir=f'EuropeLogsDemand/{NAME}')

            filepath = "eNet-{epoch:02d}-{mean_absolute_percentage_error:.3f}"
            #checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor=['mean_absolute_percentage_error'] , verbose=1, save_best_only=True, mode='min'))


            history = model.fit(train_x, train_y, batch_size=BatchSize, epochs=EPOCHS, validation_data=(validation_x,validation_y), callbacks=[tensorboard,save_best_model])

            score = model.evaluate(validation_x, validation_y, verbose=0)
            print('Test loss:', score[0])
            print('Test RMSE:', score[1])
            for s in score:
                print(s)
