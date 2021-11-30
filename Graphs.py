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
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, ConvLSTM2D, MaxPool2D, MaxPool3D, Flatten
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras import mixed_precision

from scipy import stats
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

SMALL_SIZE = 12
MEDIUM_SIZE = 15
LARGE_SIZE = 20
# df.groupby('Month')['Load'].mean().plot(figsize = (10,5))
# plt.ylabel('Energy Consumption in KWH')
# plt.ylim([0, max(df.groupby('Month')['Load'].mean()) + 10000])
# plt.xticks(df['Month'].unique())
# plt.title('Monthly Energy consumption in KWH averaged over 2 years (2018-20)')
# plt.show()



def histogram():
    plt.figure(figsize = (12,8))
    sns.distplot(df['Load'], kde=False)
    plt.title('Energy consumption (KWH) distribution over 2 years (2018-20)')
    plt.xlabel('Energy consumption in KWH')
    plt.ylabel('Days')
    # plt.rc('font', size=LARGE_SIZE)
    plt.show()
histogram()

def tempAndLoadTogether():
    # Plotting the energy and weather data on the same graph as line plots
    fig, ax1 = plt.subplots(figsize=(15, 9))
    rolling_num = 24 * 30  # smoothing the data a bit by taking the mean of last 'rolling_num' values
    # i.e. plotting the 30 day average energy consumption and temperature values
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=MEDIUM_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    color = 'tab:red'
    ax1.set_xlabel('DateTime')
    ax1.set_ylabel('Energy consumption KWH', color=color)
    ax1.plot(df['Load'].rolling(rolling_num).mean(), color=color, alpha=0.5)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Temp F', color=color, )  # we already handled the x-label with ax1
    ax2.plot(df['Temp'].rolling(rolling_num).mean(), color=color, alpha=0.5)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.suptitle('Energy Consumption and Temperature Correlation', fontsize=LARGE_SIZE)
    fig.tight_layout()
    plt.show()

tempAndLoadTogether()