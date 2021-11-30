import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, ConvLSTM2D, MaxPool2D, MaxPool3D, Flatten
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

model = Sequential()
model = load_model("models/model1.05.model")
print("Predicting...")
# array = np.array([[[0.8,0.8,1,3,5,10],[0.34,0.8,1,3,5,10],[0.34,0.8,1,3,5,10],[0.34,0.8,1,3,5,10],[0.34,0.8,1,3,5,10],[0.34,0.8,1,3,5,10],[0.34,0.8,1,3,5,10],[0.34,0.8,1,3,5,10],[0.34,0.8,1,3,5,10],[0.34,0.8,1,3,5,10],[0.34,0.8,1,3,5,10],[0.34,0.8,1,3,5,10],[0.34,0.8,1,3,5,10],[0.34,0.8,1,3,5,10],[0.34,0.8,1,3,5,10],[0.34,0.8,1,3,5,10],[0.34,0.8,1,3,5,10],[0.34,0.8,1,3,5,10],[0.34,0.8,1,3,5,10],[0.34,0.8,1,3,5,10],[0.34,0.8,1,3,5,10],[0.34,0.8,1,3,5,10],[0.34,0.8,1,3,5,10],[0.34,0.8,1,3,5,10]]])
array = np.array([[[0.4,0.2,1,3,5,5]]]) #153224
print(array.shape)
print(array)
prediction = model.predict(array)
print(prediction)

