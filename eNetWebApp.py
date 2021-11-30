import tensorflow as tf
import streamlit as st
from tensorflow.keras.models import Sequential, load_model


model = Sequential()
model = load_model("models/model1.05.model")

st.write("""
#eNet Load Data Prediction App
This app predicts load values in kw/h!
""")

st.sidebar.header("User Input Parameters")

def user_input_features():
    st.sidebar.date_input()



