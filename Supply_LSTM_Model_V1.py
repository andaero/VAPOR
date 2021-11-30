from collections import deque

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, ConvLSTM2D, MaxPool2D, MaxPool3D, Flatten, Bidirectional, RNN, InputLayer
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras import mixed_precision
from functools import reduce

from scipy import stats
from kerasncp import wirings
from kerasncp.tf import LTCCell
import seaborn as sns

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

#import in supply csv + date time
importSupplyDf = pd.read_csv("Data/DemandCharge.csv", parse_dates=["DateTime"], usecols=["DateTime", "OnCampusGeneration"], ) #15 min avg in kWatts
#import temperature + date time
#Convert 15 min increments to 1hr for supplyDf + tempDf
supplyDf = importSupplyDf.set_index("DateTime").resample('H').sum()
#remove any outliers
supplyDf = supplyDf[(np.abs(stats.zscore(supplyDf)) < 3).all(axis=1)]
print(supplyDf)

def weatherPreprocessing(filepath):
    importWeatherDf = pd.read_csv(filepath, skiprows=2, usecols=["Year","Month","Day","Hour","Minute", "Clearsky DHI", "Clearsky DNI", "Clearsky GHI", "Temperature"])

    importWeatherDf["DATE"] = pd.to_datetime(importWeatherDf[["Year","Month","Day","Hour","Minute"]])


    #Drop NaN values
    importWeatherDf = importWeatherDf.dropna()

    importWeatherDf = importWeatherDf.loc[importWeatherDf["DATE"] <= "2020-2-28"]

    weatherDf = importWeatherDf.set_index("DATE").resample('H').mean()
    return weatherDf

weatherDf2018 = weatherPreprocessing("Data/Solar_Irradiance/Solar_Irradiance_2018.csv")
weatherDf2019 = weatherPreprocessing("Data/Solar_Irradiance/Solar_Irradiance_2019.csv")
weatherDf2020 = weatherPreprocessing("Data/Solar_Irradiance/Solar_Irradiance_2020.csv")

#print(importSupplyDf.head())
#print(supplyDf.head())
#print(tempDf.head())
#print(supplyDf.head())
data_frames = [weatherDf2018,weatherDf2019,weatherDf2020]
# print(weatherDf2018.head())
weatherDf = pd.concat(data_frames)
#print("WeatherDF", weatherDf)


df = pd.merge(supplyDf, weatherDf, left_on=['DateTime'], how='outer', right_index=True)
print(df.columns)
df.dropna(inplace=True)


#print(df)





df.columns = ["Date","Supply","Year","Month","Day","Hour","Minute", "DHI", "DNI", "GHI", "Temp"]

print(df.head())
print(df.dtypes)
df = df.drop(["Date","Year","Minute"], axis=1)


# df.plot(y=["Supply", "Temp"])
# plt.show()

#Create a target column for supply in future
future = 24 #predicting 24 hours in the future
seq_len = 48 #take the last 24 hours of info

df["target"] = df["Supply"].shift(-future)
df.dropna(inplace=True)

print(df.head())

print(df.shape)


#Figuring out which parts of data to use for prediction vs results
print("Moving on to sorting data")
# Scale the nums to be between 0-1
scaler = MinMaxScaler()
df["Supply"] = scaler.fit_transform(df["Supply"].values.reshape(-1,1))
df["Temp"] = scaler.fit_transform(df["Temp"].values.reshape(-1,1))
df["DHI"] = scaler.fit_transform(df["DHI"].values.reshape(-1,1))
df["DNI"] = scaler.fit_transform(df["DNI"].values.reshape(-1,1))
df["GHI"] = scaler.fit_transform(df["GHI"].values.reshape(-1,1))


#df["target"] = scaler.fit_transform(df["Temp"].values.reshape(-1,1))

# df["Weekday"] = scaler.fit_transform(df["Weekday"].values.reshape(-1,1))
# df["Month"] = scaler.fit_transform(df["Weekday"].values.reshape(-1,1))
# df["Day"] = scaler.fit_transform(df["Weekday"].values.reshape(-1,1))
# df["Hour"] = scaler.fit_transform(df["Weekday"].values.reshape(-1,1))


# df.plot(y=["Supply", "Temp"])
# plt.show()

times = df.index.values
last_10 = df.index.values[-int(0.1*len(times))]
validation_df = df[(df.index >= last_10)]
main_df = df[(df.index< last_10)]

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

train_x, train_y = preprocess(df)
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
        self.lowestLoss = 1.3
    def on_epoch_end(self, epoch, logs=None):
        #print(logs[self.save_best_metric])
        if(logs[self.save_best_metric] < self.lowestLoss):
            self.lowestLoss = logs[self.save_best_metric]
            self.model.save(f"supply_model/{seq_len}hrinputmodel{round(logs[self.save_best_metric],2)}error.model")
            print(f"Model with MAPE of {logs[self.save_best_metric]} saved")

save_best_model = SaveBestModel()

EPOCHS = 200
BatchSizes = [128]
learning_rs = [0.0005]
layers = [3]
dense = 2;
dBatchSize = 1;

inter = 32
command_neurons = 16
wiring = wirings.NCP(
  inter_neurons=inter,  # Number of inter neurons
  command_neurons=command_neurons,  # Number of command neurons
  motor_neurons=1,  # Number of motor neurons
  sensory_fanout=10,  # How many outgoing synapses has each sensory neuron
  inter_fanout=10,  # How many outgoing synapses has each inter neuron
  recurrent_command_synapses=10,  # Now many recurrent synapses are in the
  # command neuron layer
  motor_fanin=8,  #  How many incoming syanpses has each motor neuron
)
rnn_cell = LTCCell(wiring)

def plot_wiring():
    sns.set_style("white")
    plt.figure(figsize=(12, 12))
    legend_handles = rnn_cell.draw_graph(layout='spiral', neuron_colors={"command": "tab:cyan"})
    plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()


for BatchSize in BatchSizes:
    for learning_r in learning_rs:
        for layer in layers:
            model = Sequential()
            NAME = f"Liquid-LSTM-MSE -Dense Layers-{dense}-Command Neurons-{command_neurons}-inter neurons-{inter}Filters-{BatchSize}-Layers{layer}-Learning Rate-{learning_r}-Time-{int(time.time())}"
            #Bidirectional LSTM:
            # model.add(LSTM(BatchSize, input_shape=(train_x.shape[1:]), return_sequences=True))
            # for x in range(layer - 1):
            #     model.add(Bidirectional(LSTM(BatchSize, input_shape=(train_x.shape[1:]), return_sequences=True)))
            #     model.add(Dropout(0.1))
            #     model.add(BatchNormalization())
            #LSTM:
            # for x in range(layer):
            #     model.add(LSTM(BatchSize, input_shape=(train_x.shape[1:]), return_sequences=True))
            #     model.add(Dropout(0.1))
            #     model.add(BatchNormalization())

            model.add(InputLayer(input_shape=train_x.shape[1:]))
            model.add(RNN(rnn_cell, return_sequences=True))
            model.add(Dropout(0.1))
            model.add(BatchNormalization())
            for x in range(layer):
                model.add(LSTM(BatchSize, return_sequences=True))
                model.add(Dropout(0.1))
                model.add(BatchNormalization())

            # model.add(tf.keras.Input(shape=train_x.shape[1:]))
            model.add(Flatten())
            model.add(Dense(BatchSize, activation="linear"))
            model.add(Dense(dBatchSize, activation="linear"))



            opt = tf.keras.optimizers.Adam(learning_rate= learning_r, decay=1e-6)
            model.compile(loss="mean_squared_error", optimizer=opt, metrics=[tf.keras.metrics.RootMeanSquaredError(),tf.keras.metrics.MeanAbsolutePercentageError()]) #tf.keras.metrics.RootMeanSquaredError()

            model.summary()

            tensorboard = TensorBoard(log_dir=f'SupplyLogsLiquid/{NAME}',histogram_freq=1,write_images=True)

            filepath = "eNet-{epoch:02d}-{mean_absolute_percentage_error:.3f}"
            #checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor=['mean_absolute_percentage_error'] , verbose=1, save_best_only=True, mode='max'))

            # plot_wiring()
            history = model.fit(train_x, train_y, batch_size=BatchSize, epochs=EPOCHS, validation_data=(validation_x,validation_y), callbacks=[tensorboard,save_best_model])

            score = model.evaluate(validation_x, validation_y, verbose=0)
            print('Test loss:', score[0])
            print('Test RMSE:', score[1])
            for s in score:
                print(s)

