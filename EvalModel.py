import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, ConvLSTM2D, MaxPool2D, MaxPool3D, Flatten
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from kerasncp.tf import LTCCell
import streamlit as st
import pandas as pd


st.write("# VAPOR Energy Supply and Demand Forecasting")

df = pd.read_csv("_SanDiegoCode/validation_df_supply_SD.csv", usecols=["Supply", "Month", "Day", "Hour", "DNI", "DHI", "GHI", "Temp"])
index = df.index[(df["Month"]==12) & (df["Day"]==24) & (df["Hour"]==0)] #excludes 16?
print(index[0])
input_df = df.iloc[(index[0]-48):index[0]]
print(input_df)
input2D = input_df.to_numpy()

model_input = input2D.reshape(1,input2D.shape[0],input2D.shape[1])
print(model_input.shape)
model = load_model("_SanDiegoCode/supply_model/48hrinputLiquid-600.53-RMSE.model")
prediction = model.predict(model_input)
print(prediction)