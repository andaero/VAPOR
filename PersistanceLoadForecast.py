from collections import deque

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from sklearn.preprocessing import MinMaxScaler

from scipy import stats
from sklearn.metrics import mean_squared_error



#import in load csv + date time
importLoadDf = pd.read_csv("Data/DemandCharge.csv", parse_dates=["DateTime"], usecols=["DateTime", "TotalCampusLoad"], ) #15 min avg in kWatts



#Drop NaN values

#Convert 15 min increments to 1hr for loadDf + tempDf
df = importLoadDf.set_index("DateTime").resample('H').sum()


#print(importLoadDf.head())
#print(importTempDf.dtypes)
print(df.dtypes)


#print(importLoadDf.head())
#print(loadDf.head())
#print(tempDf.head())
#print(loadDf.head())




#print(df.head())






print(df.head())
#remove any outliers
df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]

# df.plot(y=["Load", "Temp"])
# plt.show()

#Create a target column for load in future
future = 24 #predicting 24 hours in the future
seq_len = 1 #take the last 24 hours of info

df["target"] = df["TotalCampusLoad"].shift(-future)
df.dropna(inplace=True)

print(df.head())

print(df.shape)


#Figuring out which parts of data to use for prediction vs results
print("Moving on to sorting data")



# df.plot(y=["Load", "Temp"])
# plt.show()

times = df.index.values
last_10 = df.index.values[-int(0.1*len(times))]
validation_df = df[(df.index >= last_10)]
main_df = df[(df.index< last_10)]

print(validation_df)
print(main_df)


validation_x = main_df["TotalCampusLoad"].to_numpy()
validation_y = main_df["target"].to_numpy()



validation_x = np.asarray(validation_x)
validation_y = np.asarray(validation_y)




# validation_x = np.squeeze(validation_x)



def mape(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

# validation_x = np.squeeze(validation_x)

print(validation_x)
print(validation_y)
print("RMSE:", mean_squared_error(validation_y, validation_x, squared=False))

print(mape(validation_x, validation_y))
