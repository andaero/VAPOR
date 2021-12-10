import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, ConvLSTM2D, MaxPool2D, MaxPool3D, Flatten
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from kerasncp.tf import LTCCell
from kerasncp import wirings
import datetime

import streamlit as st
import pandas as pd

#streamlit run eNetWebApp.py

st.title("VAPOR Energy Supply and Demand Forecasting")
st.sidebar.header("User Input Parameters")

def user_input_features():
    date = st.sidebar.date_input("Choose a day for forecasting", value = datetime.date(2019,12,13), min_value=datetime.date(2019,12,13), max_value=datetime.date(2020,2,26))
    hour = st.sidebar.slider("Choose an hour",value=0,max_value=23)
    return date, hour

date, hour= user_input_features()



@st.cache
def getData():
    return pd.read_csv("_SanDiegoCode/validation_df_supply_SD.csv")


df = getData()
index = df.index[(df["Month"]==date.month) & (df["Day"]==date.day) & (df["Hour"]==hour)] #finds the index(row) value that corresponds w user inputted date
#Converts CSV

print(index[0])
index_df = df.iloc[(index[0]-47):(index[0]+1)] #Add 1 bc iloc is exclusive
input_df = index_df.drop(columns = ["target"])

input2D = input_df.to_numpy()

model_input = input2D.reshape(1,input2D.shape[0],input2D.shape[1])
print(model_input.shape)

@st.cache(allow_output_mutation=True)
def loadModel():
    model = load_model("_SanDiegoCode/supply_model/48hrinputLiquid-600.53-RMSE.model")
    return model
model = loadModel()
prediction = model.predict(model_input)
st.subheader(f"The supply forecast for {date} is")
st.header(f"{prediction[0][0]} KWH")
realVal = index_df.iloc[0]["target"]
st.subheader(f"The real value is")
st.header(realVal)


