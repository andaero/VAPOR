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
model = load_model("models/model1.05.model")
print("Predicting...")



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

# print(f"train data: {len(train_x)}, validation: {len(validation_x)}")

train_x = np.asarray(train_x)
train_y = np.asarray(train_y)
validation_x = np.asarray(validation_x)
validation_y = np.asarray(validation_y)

# print(train_x)
# print(train_y)
print(train_x.shape, train_y.shape, validation_x.shape, validation_y.shape)

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
    plt.ylabel("Energy Consumption(kwh)", fontsize=MEDIUM_SIZE)

    #Prediction line
    plt.plot(prediction[:100], color="darkturquoise", label="Prediction")
    #plt.legend()

    #Real line
    plt.plot(validation_y[:100], color="darkviolet", label="Real")
    plt.legend()

    plt.title("Comparison Between Predicted and Real Energy Consumption", fontsize=LARGE_SIZE)
    plt.savefig("Graphs/DemandGraphPredvReal.png")

    plt.show()

PredictionVsReal()

