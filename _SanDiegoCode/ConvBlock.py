import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, MaxPool2D, MaxPool3D, Flatten, RNN, Bidirectional, InputLayer, Conv2D, TimeDistributed, Input, MaxPooling2D, AveragePooling2D

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filter):
        super(ConvBlock, self).__init__()
        self.T_CNN_1 = TimeDistributed(Conv2D(filter, (2,2), activation="relu", padding="same"))
        self.T_CNN_2 = TimeDistributed(Conv2D(filter/2, (2,2), activation="relu", padding="same"))
        self.T_CNN_3 = TimeDistributed(Conv2D(filter/4, (2,2), activation="relu", padding="same"))
        self.T_CNN_4 = TimeDistributed(Conv2D(1, (2,2), activation="relu", padding="same"))

    def call(self, inputs):
        x = self.T_CNN_1(inputs)
        x = self.T_CNN_2(x)
        x = self.T_CNN_3(x)
        x = self.T_CNN_4(x)
        return x



