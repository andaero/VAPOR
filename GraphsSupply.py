
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
from functools import reduce
from scipy import stats
import seaborn as sns


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
df = df.drop(["Year","Minute"], axis=1)


# df.plot(y=["Supply", "Temp"])
# plt.show()


SMALL_SIZE = 12
MEDIUM_SIZE = 15
LARGE_SIZE = 20
def histogram():
    plt.figure(figsize = (12,8))
    sns.distplot(df['Supply'], kde=False)
    plt.title('Energy production (KWH) distribution over 2 years (2018-20)')
    plt.xlabel('Energy production in KWH')
    plt.ylabel('Days')
    plt.show()
histogram()


def IrradianceAndSupplyTogether():
    # Plotting the energy and weather data on the same graph as line plots
    fig, ax1 = plt.subplots(figsize=(15, 9))
    rolling_num = 1  # smoothing the data a bit by taking the mean of last 'rolling_num' values
    # i.e. plotting the 30 day average energy consumption and temperature values
    color = 'tab:red'

    ax1.set_xlabel('DateTime')
    ax1.set_ylabel('Energy production KWH', color=color)
    ax1.plot(df['Supply'].rolling(rolling_num).mean(), color=color, alpha=0.5)
    ax1.tick_params(axis='y', labelcolor=color)

    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=MEDIUM_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels

    fig.suptitle('Energy production in the UC San Diego Microgrid From 2018-2020', fontsize=LARGE_SIZE)
    fig.tight_layout()
    plt.show()

IrradianceAndSupplyTogether()

df= df.loc[df["Date"] <= "2020-1-7"]
df = df.loc[df["Date"] >= "2020-1-6"]
def SimpleIrradiance():
    fig, ax1 = plt.subplots(figsize=(15, 9))


    rolling_num = 1  # smoothing the data a bit by taking the mean of last 'rolling_num' values
    # i.e. plotting the 30 day average energy consumption and temperature values
    color = 'tab:red'
    ax1.set_xlabel('Hour')
    ax1.set_xlabel('DateTime')
    ax1.set_ylabel('Energy production KWH', color=color)
    ax1.plot(df['Supply'], color=color, alpha=0.5)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Global Horizontal Irradiance(GHI)', color=color)  # we already handled the x-label with ax1
    ax2.plot(df['GHI'], color=color, alpha=0.5)

    ax2.tick_params(axis='y', labelcolor=color)


    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=MEDIUM_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels

    fig.suptitle('Hourly Energy production and Solar Irradiance Correlation from 1/6/2021', fontsize=LARGE_SIZE)
    fig.tight_layout()
    plt.show()
SimpleIrradiance()