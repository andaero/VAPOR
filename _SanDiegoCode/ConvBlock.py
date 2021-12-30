import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, MaxPool2D, MaxPool3D, Flatten, RNN, Bidirectional, InputLayer, Conv2D, TimeDistributed, Input, MaxPooling2D, AveragePooling2D, LayerNormalization

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filter, filterSize):
        super(ConvBlock, self).__init__()
        self.T_CNN_1 = TimeDistributed(Conv2D(filter, (filterSize,filterSize), activation="relu", padding="same"))
        self.LayerNorm_1 = TimeDistributed(LayerNormalization())
        self.Dropout_1 = Dropout(0.2)


        self.T_CNN_2 = TimeDistributed(Conv2D(filter/2, (filterSize+1,filterSize+1), activation="relu", padding="same"))
        self.LayerNorm_2 = TimeDistributed(LayerNormalization())
        self.Dropout_2 = Dropout(0.2)


        self.T_CNN_3 = TimeDistributed(Conv2D(filter/4, (filterSize+2,filterSize+2), activation="relu", padding="same"))
        self.LayerNorm_3 = TimeDistributed(LayerNormalization())
        self.Dropout_3 = Dropout(0.2)

        self.T_CNN_4 = TimeDistributed(Conv2D(filter/4, (filterSize + 3, filterSize + 3), activation="relu", padding="same"))
        self.LayerNorm_4 = TimeDistributed(LayerNormalization())
        self.Dropout_4 = Dropout(0.2)

        self.T_CNN_5 = TimeDistributed(Conv2D(1, (filterSize+3,filterSize+3), activation="relu", padding="same"))

    def call(self, inputs):
        x = self.T_CNN_1(inputs)
        x = self.LayerNorm_1(x)
        x = self.Dropout_1(x)

        x = self.T_CNN_2(x)
        x = self.LayerNorm_2(x)
        x = self.Dropout_2(x)


        x = self.T_CNN_3(x)
        x = self.LayerNorm_3(x)
        x = self.Dropout_3(x)

        x = self.T_CNN_4(x)
        x = self.LayerNorm_4(x)
        x = self.Dropout_4(x)


        x = self.T_CNN_5(x)
        return x



