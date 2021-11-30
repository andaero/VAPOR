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
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, MaxPool2D, MaxPool3D, Flatten
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras import mixed_precision
from scipy import stats

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

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

#import in load csv + date time
importLoadDf = pd.read_csv("Data/DemandCharge.csv", parse_dates=["DateTime"], usecols=["DateTime", "TotalCampusLoad"], ) #15 min avg in kWatts
#import temperature + date time
importTempDf = pd.read_csv("Data/Weather.csv", parse_dates=["DATE"], usecols=["DATE", "HourlyDryBulbTemperature"])



#Drop NaN values
importTempDf = importTempDf.dropna()

#Convert 15 min increments to 1hr for loadDf + tempDf
loadDf = importLoadDf.set_index("DateTime").resample('H').sum()


#print(importLoadDf.head())
#print(importTempDf.dtypes)
tempDf = importTempDf.set_index("DATE").resample('H').mean()
print(loadDf.dtypes)


#print(importLoadDf.head())
#print(loadDf.head())
#print(tempDf.head())
#print(loadDf.head())


df = pd.merge(loadDf, tempDf, how = 'outer', left_on= 'DateTime', right_index=True)

weekdays = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3: 'Thursday', 4: 'Friday', 5:'Saturday', 6:'Sunday'}
df['Weekday'] = df.index.to_series().dt.weekday #.map(weekdays) to show which physical day it is
df['month'] = df.index.to_series().dt.month
df['day'] = df.index.to_series().dt.day
df['hour'] = df.index.to_series().dt.hour

#print(df.head())





df.columns = ["Load", "Temp", "Weekday", "Month", "Day", "Hour"]

print(df.head())
print(df.dtypes)
#remove any outliers
df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]

# df.plot(y=["Load", "Temp"])
# plt.show()

#Create a target column for load in future
future = 24 #predicting 24 hours in the future
seq_len = 48 #take the last 24 hours of info

df["target"] = df["Load"].shift(-future)
df.dropna(inplace=True)

print(df.head())

print(df.shape)


#Figuring out which parts of data to use for prediction vs results
print("Moving on to sorting data")
# Scale the nums to be between 0-1
scaler = MinMaxScaler()
df["Load"] = scaler.fit_transform(df["Load"].values.reshape(-1,1))
print("Scale is ", scaler.scale_)
df["Temp"] = scaler.fit_transform(df["Temp"].values.reshape(-1,1))
print("Scale is ", scaler.scale_)

#df["target"] = scaler.fit_transform(df["Temp"].values.reshape(-1,1))

# df["Weekday"] = scaler.fit_transform(df["Weekday"].values.reshape(-1,1))
# df["Month"] = scaler.fit_transform(df["Weekday"].values.reshape(-1,1))
# df["Day"] = scaler.fit_transform(df["Weekday"].values.reshape(-1,1))
# df["Hour"] = scaler.fit_transform(df["Weekday"].values.reshape(-1,1))


# df.plot(y=["Load", "Temp"])
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

#FOR CONVLSTM1D ONLY
train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)
validation_x = validation_x.reshape(validation_x.shape[0], validation_x.shape[1], validation_x.shape[2], 1)


#CREATING THE MODEL

class SaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self, save_best_metric='val_mean_absolute_percentage_error'):
        self.save_best_metric = save_best_metric
        self.lowestLoss = 1.05
    def on_epoch_end(self, epoch, logs=None):
        #print(logs[self.save_best_metric])
        if(logs[self.save_best_metric] < self.lowestLoss):
            self.lowestLoss = logs[self.save_best_metric]
            self.model.save(f"models/model{round(logs[self.save_best_metric],2)}.model")
            print(f"Model with MAPE of {logs[self.save_best_metric]} saved")

save_best_model = SaveBestModel()
EPOCHS = 100
BatchSizes = [270]
learning_rs = [0.0005, 0.0005]
layers = [2]
dense = 2;
dBatchSize = 1;
in_shape = (train_x.shape[1], train_x.shape[2], 1)
print(in_shape)

for BatchSize in BatchSizes:
    for learning_r in learning_rs:
        for layer in layers:
            model = Sequential()
            NAME = f"V5-Dense Layers-{dense}-Dense Batch-{dBatchSize}-SeqLen-{seq_len}Filters-{BatchSize}-Layers{layer}-Learning Rate-{learning_r}-Time-{int(time.time())}"


            for x in range(layer-1):
                model.add(tf.keras.layers.ConvLSTM1D(BatchSize, kernel_size=12, input_shape=in_shape, return_sequences=True, data_format='channels_last', padding='same'))
                model.add(Dropout(0.1))
                model.add(BatchNormalization())

            #model.add(tf.keras.layers.ConvLSTM1D(BatchSize, kernel_size=12, input_shape=in_shape, return_sequences=True, data_format='channels_last', padding='same'))
            model.add(tf.keras.layers.ConvLSTM1D(BatchSize, kernel_size=12, return_sequences=False, data_format='channels_last', padding='same'))
            model.add(Dropout(0.1))
            model.add(BatchNormalization())

            # for x in range(layer-1):
            #     model.add(LSTM(BatchSize, input_shape=(train_x.shape[1:]), return_sequences= True))
            #     model.add(Dropout(0.1))
            #     model.add(BatchNormalization())
            #model.add(LSTM(BatchSize, input_shape=(train_x.shape[1:]), return_sequences=False))
            #model.add(tf.keras.layers.ConvLSTM1D(BatchSize, kernel_size=1, return_sequences=True, data_format='channels_last', padding='same'))

            # model.add(Dropout(0.1))
            # model.add(BatchNormalization())

            #PURE BASIC ANN - 3.7% MAPE
            # model.add(tf.keras.Input(shape=train_x.shape[1:]))
            model.add(Flatten())
            #model.add(Dense(BatchSize, activation="linear"))
            model.add(Dense(dBatchSize, activation="linear"))



            opt = tf.keras.optimizers.Adam(learning_rate= learning_r, decay=1e-6)
            model.compile(loss=tf.keras.losses.MeanAbsolutePercentageError(), optimizer=opt, metrics=[tf.keras.metrics.RootMeanSquaredError(),tf.keras.metrics.MeanAbsolutePercentageError()]) #tf.keras.metrics.RootMeanSquaredError()

            model.summary()

            tensorboard = TensorBoard(log_dir=f'CNNLSTM_Load_Logs/{NAME}', histogram_freq=1, write_images=True)

            filepath = "eNet-{epoch:02d}-{mean_absolute_percentage_error:.3f}"
            #checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor=['mean_absolute_percentage_error'] , verbose=1, save_best_only=True, mode='min'))


            history = model.fit(train_x, train_y, batch_size=BatchSize, epochs=EPOCHS, validation_data=(validation_x,validation_y), callbacks=[tensorboard,save_best_model])

            score = model.evaluate(validation_x, validation_y, verbose=0)
            print('Test loss:', score[0])
            print('Test RMSE:', score[1])
            for s in score:
                print(s)

