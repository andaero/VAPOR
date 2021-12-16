from collections import deque
import glob
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from scipy import stats
import seaborn as sns

def weatherPreprocessing(filepath):
    importWeatherDf = pd.read_csv(filepath, skiprows=2, usecols=["Year","Month","Day","Hour","Minute", "Clearsky DHI", "Clearsky DNI", "Clearsky GHI", "Temperature"])

    importWeatherDf["DATE"] = pd.to_datetime(importWeatherDf[["Year","Month","Day","Hour","Minute"]])


    #Drop NaN values
    importWeatherDf = importWeatherDf.dropna()

    importWeatherDf = importWeatherDf.loc[importWeatherDf["DATE"] <= "2020-2-28"]

    weatherDf = importWeatherDf.set_index("DATE").resample('H').mean()
    return weatherDf

def weatherPreprocessingSolcast(filepath):
    importWeatherDf = pd.read_csv(filepath, skiprows=10, usecols=["Year","Month","Day","Hour","Minute", "Cloudopacity", "DHI", "DNI", "GHI", "Tamb"])

    importWeatherDf["DATE"] = pd.to_datetime(importWeatherDf[["Year","Month","Day","Hour","Minute"]])


    #Drop NaN values
    importWeatherDf = importWeatherDf.dropna()

    importWeatherDf = importWeatherDf.loc[(importWeatherDf["DATE"] <= "2020-2-28") & (importWeatherDf["DATE"] >= "2018-1-1")]

    weatherDf = importWeatherDf.set_index("DATE")
    return weatherDf

def scaleData(df):
    scaler = MinMaxScaler()
    df["Supply"] = scaler.fit_transform(df["Supply"].values.reshape(-1,1))
    df["Temp"] = scaler.fit_transform(df["Temp"].values.reshape(-1,1))
    df["DHI"] = scaler.fit_transform(df["DHI"].values.reshape(-1,1))
    df["DNI"] = scaler.fit_transform(df["DNI"].values.reshape(-1,1))
    df["GHI"] = scaler.fit_transform(df["GHI"].values.reshape(-1,1))


    #df["target"] = scaler.fit_transform(df["Temp"].values.reshape(-1,1))

    df["Weekday"] = scaler.fit_transform(df["Weekday"].values.reshape(-1,1))
    df["Month"] = scaler.fit_transform(df["Month"].values.reshape(-1,1))
    df["Day"] = scaler.fit_transform(df["Day"].values.reshape(-1,1))
    df["Hour"] = scaler.fit_transform(df["Hour"].values.reshape(-1,1))

def scaleDataV2(df,columns):
    scaler = MinMaxScaler()
    for column in columns:
        df[column] = scaler.fit_transform(df[column].values.reshape(-1,1))

def preprocess(df,seq_len):

    sequential_data = []  # this is a list that will CONTAIN the sequences
    prev_days = deque(maxlen=seq_len)  # Actual seq made with deque, keeps the maximum length by popping out older values as new ones come in

    for i in df.to_numpy():  # iterate over the values
        prev_days.append([n for n in i[:-1]])  # store all but the target
        if len(prev_days) == seq_len:  # make sure we have 48 seq
            sequential_data.append([np.array(prev_days), i[-1]])
    random.shuffle(sequential_data)  # shuffle
    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), y

def split_main_validation_df(df):
    times = df.index.values
    last_10 = df.index.values[-int(0.1*len(times))]
    validation_df = df[(df.index >= last_10)]
    main_df = df[(df.index< last_10)]
    return main_df,validation_df

def plot_energy_gen_and_GHI(df):
    fig, ax1 = plt.subplots(figsize=(15, 9))


    color = 'tab:red'
    ax1.set_xlabel('Hour')
    ax1.set_xlabel('DateTime')
    ax1.set_ylabel('Energy production KWH', color=color)
    ax1.plot(df.iloc[9000:9100]['Supply'], color=color, alpha=0.5)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Target(GHI)', color=color)  # we already handled the x-label with ax1
    ax2.plot(df.iloc[9000:9100]['target'], color=color, alpha=0.5)

    ax2.tick_params(axis='y', labelcolor=color)

    fig.suptitle('Hourly Energy production and Solar Irradiance Correlation from 1/6/2021')
    fig.tight_layout()
    plt.show()

def df_to3D(df):
  df = df.drop(["DateTime"], axis=1)

  npy2D = df.to_numpy()
  print(npy2D.shape)
  npy3D = npy2D.reshape(-1, 10, 10)
  print(npy3D.shape)
  ax = sns.heatmap(npy3D[15])

  # plt.title("How to visualize (plot) \n a numpy array in python using seaborn ?",fontsize=12)

  # plt.savefig("visualize_numpy_array_01.png", bbox_inches='tight', dpi=100)

  plt.show()
  ax = sns.heatmap(npy3D[16])
  plt.show()

  return npy3D

def model_preprocess(seq_len):

    #import in supply csv + date time

    importSupplyDf = pd.read_csv("../Data/supplyDatav3.csv", parse_dates=["DateTime"])
    column_list = list(importSupplyDf)
    column_list.remove("DateTime")

    importSupplyDf["SupplyTotal"] = importSupplyDf[column_list].sum(axis=1) #add all rows except datetime
    # print(importSupplyDf)
    supplyDf = importSupplyDf[["DateTime","SupplyTotal"]].copy()

    # print(supplyDf)

    weatherDf2018 = weatherPreprocessing("../Data/Solar_Irradiance/Solar_Irradiance_2018.csv")
    weatherDf2019 = weatherPreprocessing("../Data/Solar_Irradiance/Solar_Irradiance_2019.csv")
    weatherDf2020 = weatherPreprocessing("../Data/Solar_Irradiance/Solar_Irradiance_2020.csv")

    data_frames = [weatherDf2018,weatherDf2019,weatherDf2020]
    # print(weatherDf2018.head())
    weatherDf = pd.concat(data_frames)
    #print("WeatherDF", weatherDf)


    df = pd.merge(supplyDf, weatherDf, left_on=['DateTime'], how='outer', right_index=True)
    df = df.dropna()


    #print(df)





    df.columns = ["Date","Supply","Year","Month","Day","Hour","Minute", "DHI", "DNI", "GHI", "Temp"]
    df['Weekday'] = df["Date"].dt.weekday

    # print(df.head())
    # print(df.dtypes)
    df = df.drop(["Date","Year","Minute"], axis=1)

    """Remove outliers"""
    df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
    # df = df[(np.abs(stats.zscore(df["Supply"])) < 3)]



    #Create a target column for supply in future
    future = 24 #predicting 24 hours in the future

    df["target"] = df["Supply"].shift(-future)
    df = df.dropna()

    # print(df.head())
    #
    # print(df.shape)


    #Figuring out which parts of data to use for prediction vs results
    print("Moving on to sorting data...")

    """---Scale the nums to be between 0-1---"""
    scaleData(df)


    #Creating main_df and validation_df
    main_df, validation_df = split_main_validation_df(df)
    # print(validation_df)
    # print(main_df)

    main_df = main_df.dropna()
    validation_df = validation_df.dropna()

    """Export dataframes to csv"""
    # main_df.to_csv("main_df_scaled_v4.csv", index=False)
    # validation_df.to_csv("validation_df_scaled_v4.csv", index=False)
    # print("Exported main_df and validation_df successfully")

    """Split into numpy arrays"""
    train_x, train_y = preprocess(main_df,seq_len)

    #np.savetxt('train_x_solar.csv', train_x, delimiter=',')
    validation_x, validation_y = preprocess(validation_df,seq_len)

    print(f"train data: {len(train_x)}, validation: {len(validation_x)}")

    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)
    validation_x = np.asarray(validation_x)
    validation_y = np.asarray(validation_y)

    # print(train_x[:1])
    # print(train_y[:1])
    print(train_x.shape, train_y.shape, validation_x.shape, validation_y.shape)
    #Replacing any nan values w 0
    train_x[np.isnan(train_x)] = 0
    train_y[np.isnan(train_y)] = 0
    validation_x[np.isnan(validation_x)] = 0
    validation_y[np.isnan(validation_y)] = 0


    return train_x,train_y,validation_x,validation_y


def model_preprocess_CNN(seq_len):

    #import in supply csv + date time

    importSupplyDf = pd.read_csv("../Data/supplyDatav4.csv", parse_dates=["DateTime"])
    print(importSupplyDf)
    column_list = ["RealPower0", "RealPower1", "RealPower32"]

    importSupplyDf["RealPower_Mod"] = importSupplyDf[column_list].sum(axis=1) #add all rows except datetime
    importSupplyDf["RealPower_42"] = 0.35*importSupplyDf["RealPower"]
    importSupplyDf["RealPower_43"] = 0.4*importSupplyDf["RealPower"]
    importSupplyDf["RealPower_44"] = 0.25*importSupplyDf["RealPower"]
    importSupplyDf["RealPower_45"] = 0.4*importSupplyDf["RealPower4"]
    importSupplyDf["RealPower_46"] = 0.6*importSupplyDf["RealPower4"]
    importSupplyDf["RealPower_47"] = 0.4*importSupplyDf["RealPower20"]
    importSupplyDf["RealPower_48"] = 0.6*importSupplyDf["RealPower20"]

    supplyDf = importSupplyDf.drop(["RealPower0", "RealPower1", "RealPower32", "RealPower","RealPower4","RealPower20",], axis=1)



    # print(importSupplyDf)
    # supplyDf = importSupplyDf.drop(column_list, axis=1)
    print(supplyDf.columns)
    print(supplyDf)
    #Preprocess weather df
    weatherDf = weatherPreprocessingSolcast("../Data/Solar_Irradiance/Solcast_Weather.csv")

    #Make PV values 2D
    npy3D = df_to3D(supplyDf)

    df = pd.merge(supplyDf, weatherDf, left_on=['DateTime'], how='outer', right_index=True)
    df = df.dropna()
    print(df.head(18))

    #Remove outliers
    df = df.drop(["DateTime","Year","Minute",], axis=1)

    df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
    print(df.columns)
    #--Normalize data--
    # normalizeList = list(df.columns)
    # scaleDataV2(df,normalizeList)
    # print(df.head(18))


