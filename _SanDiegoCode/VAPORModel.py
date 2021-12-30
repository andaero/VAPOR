import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, MaxPool2D, MaxPool3D, Flatten, RNN, Bidirectional, InputLayer, Conv2D, TimeDistributed, Input, MaxPooling2D, AveragePooling2D, LayerNormalization

from SLAM_Layer import SLAM_Layer
from ConvBlock import ConvBlock
import numpy as np

class VAPOR_Model(tf.keras.Model):
    def __init__(self, SLAM_Dense, ConvBlock_Size, tensorLen, filterSize):
        super(VAPOR_Model, self).__init__()
        self.slam_1 = SLAM_Layer(dense=SLAM_Dense, tensorLen=tensorLen)
        self.T_CNN_1 = ConvBlock(ConvBlock_Size, filterSize)
        self.Dropout_1 = Dropout(0.2)

        self.slam_2 = SLAM_Layer(dense=SLAM_Dense, tensorLen=tensorLen)
        self.T_CNN_2 = TimeDistributed(Conv2D(128, (filterSize,filterSize), activation="relu"))
        self.Dropout_2 = Dropout(0.2)

        # self.T_CNN_3 = TimeDistributed(Conv2D(256, (2,2), activation="relu"))


        self.AvgPool2D = TimeDistributed(AveragePooling2D(pool_size=(filterSize,filterSize)))
        self.T_Flatten = TimeDistributed(Flatten())

        self.Dense_1 = TimeDistributed(Dense(64, activation="relu"))
        # self.Layer_Norm = LayerNormalization(axis=[1,2,3])
        self.Layer_Norm = LayerNormalization()

        self.LSTM_1 = LSTM(128, return_sequences=True)
        self.Layer_Norm_1 = LayerNormalization()

        self.LSTM_2 = LSTM(128, return_sequences=True)
        self.Layer_Norm_2 = LayerNormalization()

        self.LSTM_3 = LSTM(128, return_sequences=False)
        self.Layer_Norm_3 = LayerNormalization()


        self.Flatten_LSTM = Flatten()
        self.Dense_LSTM_1 = Dense(64, activation="linear")

        self.Dense_LSTM_2 = Dense(1, activation="linear")

        # self.main = MainModel()

    def call(self, inputs):
        x_main, x_aux = inputs
        x = self.slam_1(x_main, x_aux)
        x = self.T_CNN_1(x)
        x = self.Layer_Norm(x)

        x = self.Dropout_1(x)


        x = self.slam_2(x, x_aux)
        x = self.T_CNN_2(x)
        x = self.Dropout_2(x)

        # x = self.T_CNN_3(x)

        x = self.AvgPool2D(x)
        x = self.T_Flatten(x)
        x = self.Dense_1(x)

        x = self.LSTM_1(x)
        x = self.Layer_Norm_1(x)
        x = self.LSTM_2(x)
        x = self.Layer_Norm_2(x)
        x = self.LSTM_3(x)
        x = self.Layer_Norm_3(x)

        x = self.Flatten_LSTM(x)
        x = self.Dense_LSTM_1(x)
        x = self.Dense_LSTM_2(x)

        return x