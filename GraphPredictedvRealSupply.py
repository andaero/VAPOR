import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

import tensorflow as tf

import random
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

from tensorflow.keras.models import Sequential, load_model


#Load Code
model = Sequential()
model = load_model("supply_model/48hrinputmodel1.13error.model")
print("Preprocessing...")


#import in supply csv + date time
importSupplyDf = pd.read_csv("Data/DemandCharge.csv", parse_dates=["DateTime"], usecols=["DateTime", "OnCampusGeneration"], ) #15 min avg in kWatts
#import temperature + date time
#Convert 15 min increments to 1hr for supplyDf + tempDf
supplyDf = importSupplyDf.set_index("DateTime").resample('H').sum()
#remove any outliers
supplyDf = supplyDf[(np.abs(stats.zscore(supplyDf)) < 3).all(axis=1)]
# print(supplyDf)

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

# print(df.head())
print(df.dtypes)
df = df.drop(["Date","Year","Minute"], axis=1)


# df.plot(y=["Supply", "Temp"])
# plt.show()

#Create a target column for supply in future
future = 24 #predicting 24 hours in the future
seq_len = 48 #take the last 24 hours of info

df["target"] = df["Supply"].shift(-future)
df.dropna(inplace=True)

# print(df.head())

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

# print(validation_df)
# print(main_df)

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


#MODEL PREDICTION
prediction = model.predict(validation_x)
for x in range(10):
    print(f"Prediction:{prediction[x]} Real:{validation_y[x]}")


SMALL_SIZE = 10
MEDIUM_SIZE = 13
LARGE_SIZE = 15
def PredictionVsReal():
    plt.figure(figsize=(8,5))
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels

    plt.xlabel("Time(hours)", fontsize=MEDIUM_SIZE)
    plt.ylabel("Energy Generation(kwh)", fontsize=MEDIUM_SIZE)

    #plt.legend()



    #Prediction line
    plt.plot(prediction[:100], color="darkturquoise", label="Prediction")

    # Real line
    plt.plot(validation_y[:100], color="darkorange", label="Real")

    plt.legend()

    plt.title("Comparison Between Predicted and Real Energy Generation", fontsize=LARGE_SIZE)
    plt.savefig("Graphs/SupplyGraphPredvReal.png")

    plt.show()

PredictionVsReal()