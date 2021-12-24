from collections import deque
import glob

import numpy
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.utils import shuffle
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

    importWeatherDf = importWeatherDf.loc[(importWeatherDf["DATE"] <= "2020-2-29") & (importWeatherDf["DATE"] >= "2018-1-1")]

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

def preprocess_aux_data(df,seq_len,supplyTotal):
    """Converts aux data from df into groups of 3 vectors"""

    """ NEED 4 HOURS FOR EACH SEQ LEN"""
    if(supplyTotal==False):
        df = df.drop(["SupplyTotal"], axis=1)
    columnNum = len(df.columns) - 1 # SUBTRACT 1 FOR TARGET VALUE
    rowValue = int(seq_len/4) #For seq len of 12 would be 3
    prev_days = deque(maxlen=(seq_len))  # Actual seq made with deque, keeps the maximum length by popping out older values as new ones come in
    # print(seq_len*columnNum)
    sequential_data = np.empty([(len(df.index)-seq_len+1),rowValue, columnNum*4])  # this is a list that will CONTAIN the sequences np.empty([])
    target_data = []
    """ WORKS NOW"""

    n = 0
    for i in df.to_numpy():  # iterate over the values
        prev_days.append([n for n in i[:-1]])  # store all but the target
        if len(prev_days) == seq_len:  # make sure we have 60 numbers
            MatrixGroup = np.array(prev_days).reshape(rowValue, columnNum*4)
            """ columnNum*4 = [temp, opac, DHI, DNI, GHI]*4 for each column vector row
                rowValue = length of matrix to match up with the # of PV gen matrices
            """
            sequential_data[n] = MatrixGroup # add all values as a 2d array
            # sequential_data.append(MatrixGroup)
            target_data.append(i[-1])
            n+=1
    # print(sequential_data.shape)
    target_np = np.array(target_data)
    # print(target_np.shape)
    # print(sequential_data[2])


    sequential_data, target_np = shuffle(sequential_data,target_np,random_state=100)

    return sequential_data, target_np


def df_to3D(df, seq_len, show_fig):
    """Converts df to 5x5 PV gen values, then into groups of 3 10x10 PV gen matrices"""
    # df = df.drop(["DateTime"], axis=1)
    seq_len_divided_4 = int(seq_len/4)
    npy2D = df.to_numpy()
    print(npy2D.shape)
    npy3D = npy2D.reshape(-1, 5, 5)
    length = int(npy3D.shape[0])
    #MAKE THSE INTO
    sequential_data = np.empty([length-seq_len+1,seq_len_divided_4,10,10])  # numpy array contains the sequences without accounting for the dimensionality
    for t in range(int(length-seq_len+1)):
        x = 0
        series = np.empty([3, 10, 10])
        for i in range(0,seq_len_divided_4,4):
            series[x]= np_FiveToTen(t+i, npy3D)
            x+=1
        sequential_data[t] = series
    # npy3D = npy2D.reshape(-1, 10, 10)
    # print(npy3D.shape)
    print(sequential_data.shape)

    sequential_data = shuffle(sequential_data,random_state=42)

    for map2D in sequential_data[23]:
        ax = sns.heatmap(map2D)
        plt.title("How to visualize (plot) \n a numpy array in python using seaborn",fontsize=12)
        plt.savefig("visualize_numpy_array.png", bbox_inches='tight', dpi=100)
        if(show_fig):
            plt.show()



    return sequential_data

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

def np_FiveToTen(t, npy3D):
    npy2DUpper = np.concatenate([npy3D[t], npy3D[t + 1]], axis=1)
    npy2DLower = np.concatenate([npy3D[t + 2], npy3D[t + 3]], axis=1)
    npTenByTen = np.concatenate([npy2DUpper, npy2DLower], axis=0)
    return npTenByTen



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


def model_preprocess_CNN(seq_len, supplyTotal, showFig):
    "Preprocesses data into groups of 3 10x10 PV matrices and aux data vectors, supplyTotal=True if include in auxDf"
    #import in supply csv + date time

    importSupplyDf = pd.read_csv("../Data/supplyDatav4.csv", parse_dates=["DateTime"])
    # print(importSupplyDf)

    importSupplyDf["RealPower_42"] = 0.35*importSupplyDf["RealPower"]
    importSupplyDf["RealPower_43"] = 0.4*importSupplyDf["RealPower"]
    importSupplyDf["RealPower_44"] = 0.25*importSupplyDf["RealPower"]
    importSupplyDf["RealPower_45"] = 0.4*importSupplyDf["RealPower4"]
    importSupplyDf["RealPower_46"] = 0.6*importSupplyDf["RealPower4"]
    importSupplyDf["RealPower_47"] = 0.4*importSupplyDf["RealPower20"]
    importSupplyDf["RealPower_48"] = 0.6*importSupplyDf["RealPower20"]

    supplyDf = importSupplyDf.drop(["RealPower0", "RealPower32", "RealPower","RealPower4","RealPower20"], axis=1)


    # print(supplyDf)
    # print(importSupplyDf)
    # supplyDf = importSupplyDf.drop(column_list, axis=1)

    print(supplyDf.columns)
    # print(supplyDf)

    column_list = list(importSupplyDf)
    column_list.remove("DateTime")
    importSupplyDf["SupplyTotal"] = importSupplyDf[column_list].sum(axis=1)  # add all rows except datetime
    # print(importSupplyDf)
    supplyTotalDf = importSupplyDf[["DateTime", "SupplyTotal"]].copy()


    #Preprocess weather df
    weatherDf = weatherPreprocessingSolcast("../Data/Solar_Irradiance/Solcast_Weather.csv")


    # npy4D = npy3D.reshape(-1,3,10,10) - MAKE SURE THAT IT ALSO USES DEQUE

    supplyDfColumns = list(supplyDf.columns)
    supplyDfColumns.remove("DateTime")
    supplyDf = supplyDf.set_index("DateTime")
    #Double check to sync PV gen data with weather data
    print(supplyTotalDf)
    df = pd.merge(supplyTotalDf, weatherDf, left_on=['DateTime'], how='outer', right_index=True)
    df = pd.merge(df, supplyDf, left_on=['DateTime'], how='outer', right_index=True)
    df = df.dropna()
    # Create a target column for supply in future
    future = 24  # predicting 24 hours in the future

    df["target"] = df["SupplyTotal"].shift(-future)
    df = df.dropna()

    # REMOVE ALL OUTLIERS
    df = df.drop(["DateTime","Year","Month","Day","Hour","Minute"],axis=1)

    print(df.columns)
    # print(df)

    df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]

    # pvDf = df.drop([])

    #PV GEN STUFF
    pvDf = df.drop(["SupplyTotal", "target"], axis=1)
    # print(supplyDfColumns)
    pvDf = pvDf.drop(["Cloudopacity", "DHI", "DNI", "GHI", "Tamb"], axis=1)
    print("COLUMNS FOR PV:", pvDf.columns)



    #AUX DF STUFF
    auxDf = df.drop(supplyDfColumns, axis=1)


    #Remove outliers

    # Want preprocessing to do the sequencing for just aux outputs + target

    """PV DATA SPLIT + PREPROCESSING"""
    main_df_pv, validation_df_pv = split_main_validation_df(pvDf)

    train_x_pv = df_to3D(main_df_pv, seq_len, show_fig=showFig)
    validation_x_pv = df_to3D(validation_df_pv, seq_len, show_fig=showFig)

    print("Train X Shape PV: ", train_x_pv.shape)

    print("Validation X Shape PV:  ", validation_x_pv.shape)

    """AUX DATA SPLIT + PREPROCESSING"""
    main_df_aux, validation_df_aux = split_main_validation_df(auxDf)
    print("AUX DF ", len(main_df_aux.index))
    print("PV DF: ", len(main_df_pv.index))
    print(auxDf.columns)
    train_x_aux, train_y = preprocess_aux_data(main_df_aux, seq_len, supplyTotal=supplyTotal)
    validation_x_aux, validation_y = preprocess_aux_data(validation_df_aux, seq_len, supplyTotal=supplyTotal)


    #Change from just using 1 hr per vector to 4 hours

    print(train_x_aux.shape, train_y.shape, validation_x_aux.shape, validation_y.shape)


    # auxDf.to_numpy
    #--Normalize data--
    # normalizeList = list(df.columns)
    # scaleDataV2(df,normalizeList)
    # print(df.head(18))

    return train_x_pv, validation_x_pv, train_x_aux, validation_x_aux, train_y, validation_y


