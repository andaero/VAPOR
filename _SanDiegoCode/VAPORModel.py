import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow_addons.rnn import LayerNormLSTMCell
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, MaxPool2D, MaxPool3D, Flatten, RNN, Bidirectional, InputLayer, Conv2D, TimeDistributed, Input, AveragePooling2D, LayerNormalization

from SLAM_Layer import SLAM_Layer
from ConvBlock import ConvBlock, ResBlock
import numpy as np

class VAPOR_Model(tf.keras.Model):
    def __init__(self, SLAM_Dense, ConvBlock_Size, CNN_2, filterSize, CNN_Dense, tensorLen, Conv2Bool):
        super(VAPOR_Model, self).__init__()
        self.Conv2Bool = Conv2Bool

        self.slam_1 = SLAM_Layer(dense=SLAM_Dense, tensorLen=tensorLen, relu=False)
        self.T_CNN_1 = ConvBlock(ConvBlock_Size, filterSize)
        # self.T_CNN_1 = TimeDistributed(Conv2D(1, (filterSize,filterSize), activation="relu", padding="same"))

        # self.Layer_Norm = LayerNormalization()
        self.Dropout_1 = Dropout(0.1)

        self.slam_2 = SLAM_Layer(dense=SLAM_Dense, tensorLen=tensorLen, relu=False)
        self.T_CNN_2 = TimeDistributed(Conv2D(CNN_2, (filterSize,filterSize), activation="relu", padding="same"))
        # self.T_CNN_2 = ResBlock(n_filters= CNN_2, filterSize=filterSize)
        self.Dropout_2 = Dropout(0.1)


        self.AvgPool2D = TimeDistributed(AveragePooling2D(pool_size=(filterSize,filterSize)))

        self.T_Flatten = TimeDistributed(Flatten())

        self.Dense_1 = TimeDistributed(Dense(CNN_Dense, activation="linear"))

        self.LSTM_1 = LSTM(CNN_Dense, return_sequences=True, recurrent_dropout=0)
        # self.Layer_Norm_1 = LayerNormalization()

        self.LSTM_2 = LSTM(128, return_sequences=True, recurrent_dropout=0)
        # self.Layer_Norm_2 = LayerNormalization()


        self.LSTM_3 = LSTM(128, return_sequences=False, recurrent_dropout=0)
        # self.Layer_Norm_3 = LayerNormalization()


        self.Flatten_LSTM = Flatten()
        self.Dense_LSTM_1 = Dense(128, activation="linear")

        self.Dense_LSTM_2 = Dense(1, activation="linear")

        # self.main = MainModel()

    def call(self, inputs):
        x_main, x_aux = inputs
        x = self.slam_1(x_main, x_aux)
        x = self.T_CNN_1(x)
        # x = self.Layer_Norm(x)
        # x = self.Dropout_1(x)
        #
        #
        # x = self.AvgPool2D_0(x)
        x = self.slam_2(x, x_aux)
        if self.Conv2Bool == True:
            x = self.T_CNN_2(x)
            x = self.Dropout_2(x)
        #
        #
        # # x = self.T_CNN_3(x)

        x = self.AvgPool2D(x)
        x = self.T_Flatten(x)
        x = self.Dense_1(x)

        x = self.LSTM_1(x)
        # x = self.Layer_Norm_1(x)
        x = self.LSTM_2(x) #--RMSE 548.52 uses this
        # x = self.Layer_Norm_2(x)
        x = self.LSTM_3(x)
        # x = self.Layer_Norm_3(x)
        print(x)

        # x = self.rnn_1(x)
        # x = self.rnn_2(x)
        # x = self.rnn_3(x)


        x = self.Flatten_LSTM(x)
        x = self.Dense_LSTM_1(x)
        x = self.Dense_LSTM_2(x)
        print("_________TEST")

        return x

    """def call(self, inputs):
        x_main, x_aux = inputs
        # x = tf.expand_dims(x_main, axis=0)

        # x = self.T_CNN_1(x)
        # x = self.Layer_Norm(x)
        # x = self.Dropout_1(x)
        # x = self.T_CNN_2(x)

        # x = self.AvgPool2D(x)
        x = self.T_Flatten(x_main)
        # # x = self.Dense_1(x)
        #
        # x = self.LSTM_1(x)
        # # x = self.Layer_Norm_1(x)
        # x = self.LSTM_2(x)
        # # x = self.Layer_Norm_2(x)
        # x = self.LSTM_3(x)
        # # x = self.Layer_Norm_3(x)


        x = self.Flatten_LSTM(x)
        x = self.Dense_LSTM_1(x)
        x = self.Dense_LSTM_2(x)
        return x"""