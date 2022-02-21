import math
from collections import deque

import numpy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import shuffle

from scipy import stats


#import in supply csv + date time

importSupplyDf = pd.read_csv("../Data/supplyDatav4.csv", parse_dates=["DateTime"])
column_list = list(importSupplyDf)
column_list.remove("DateTime")

importSupplyDf["SupplyTotal"] = importSupplyDf[column_list].sum(axis=1) #add all rows except datetime
# print(importSupplyDf)
df = importSupplyDf[["SupplyTotal"]].copy()


print(df.head())
#remove any outliers
df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]

# df.plot(y=["Load", "Temp"])
# plt.show()

#Create a target column for load in future
future = 25 #predicting 24 hours in the future
seq_len = 1 #take the last 24 hours of info

df["target"] = df["SupplyTotal"].shift(-future)
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

validation_x = validation_df["SupplyTotal"].to_numpy()
validation_y = validation_df["target"].to_numpy()



validation_x = np.asarray(validation_x)
validation_y = np.asarray(validation_y)


validation_x = shuffle(validation_x, random_state=42)
validation_y = shuffle(validation_y, random_state=42)



validation_x = numpy.squeeze(validation_x)


print("RMSE:", math.sqrt(mean_squared_error(validation_y, validation_x)))
print("Coefficient of Determination (r^2): ", r2_score(validation_y, validation_x))
