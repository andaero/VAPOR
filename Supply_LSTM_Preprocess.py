from collections import deque
import glob
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, ConvLSTM2D, MaxPool2D, MaxPool3D, Flatten, Bidirectional
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras import mixed_precision
from functools import reduce

from scipy import stats


#import in supply csv + date time

# specifying the path to csv files
path = "Data/PVGenerator"

# csv files in the path
files = glob.glob(path + "/*.csv")
importSupplyDf = pd.read_csv("Data/Template.csv", parse_dates=["DateTime"], usecols=["DateTime", "RealPower"])

for file in files:
    print(file)
    df = pd.read_csv(file, parse_dates=["DateTime"], usecols=["DateTime", "RealPower"])
    df.fillna(method="ffill",inplace=True)
    print(df)
    # importSupplyDf["RealPower"] = importSupplyDf["RealPower"] + df["RealPower"]
    #importSupplyDf.fillna(0,inplace=True)
    importSupplyDf = importSupplyDf.merge(df,on=["DateTime"],how="left",suffixes=("_1",False))
    importSupplyDf.fillna(method="ffill",inplace=True)
    print(importSupplyDf)

print(importSupplyDf.head(10))

#importSupplyDf = pd.read_csv("Data/DemandCharge.csv", parse_dates=["DateTime"], usecols=["DateTime", "RealPower"], ) #15 min avg in kWatts
#import temperature + date time
#Convert 15 min increments to 1hr for supplyDf + tempDf
supplyDf = importSupplyDf.set_index("DateTime").resample('H').sum()
#remove any outliers
#supplyDf = supplyDf[(np.abs(stats.zscore(supplyDf)) < 3).all(axis=1)]
print(supplyDf)
supplyDf.to_csv("supplyDatav2.csv")