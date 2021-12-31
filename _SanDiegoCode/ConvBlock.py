import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, MaxPool2D, MaxPool3D, Flatten, RNN, Bidirectional, InputLayer, Conv2D, TimeDistributed, Input, MaxPooling2D, AveragePooling2D, LayerNormalization, Add, Activation

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, n_filters, filterSize):
        super(ResBlock, self).__init__()
        self.n_filters = n_filters
        self.merge_input = TimeDistributed(Conv2D(n_filters, (1, 1), padding='same', activation='relu'))
        self.conv1 = TimeDistributed(Conv2D(n_filters, (filterSize,filterSize), activation="relu", padding="same"))
        self.conv2 = TimeDistributed(Conv2D(n_filters, (filterSize,filterSize), activation="relu", padding="same"))
        self.LayerNorm_1 = TimeDistributed(LayerNormalization())
        self.LayerNorm_2 = TimeDistributed(LayerNormalization())
        self.add = TimeDistributed(Add())


    def call(self, layer_in):
        merge_input = layer_in
        # check if the number of filters needs to be increased, assumes channels last format
        if layer_in.shape[-1] != self.n_filters:
            merge_input = self.merge_input(layer_in) #kernel_initializer='he_normal' - NOT THE DEFAULT
        x = self.conv1(layer_in)
        x = self.LayerNorm_1(x)
        # x = self.conv2(x)
        # x = self.LayerNorm_2(x)

        layer_out = self.add([x, merge_input])
        layer_out = Activation("relu")(layer_out)
        return layer_out


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filter, filterSize):
        super(ConvBlock, self).__init__()
        self.T_CNN_1 = TimeDistributed(Conv2D(filter, (filterSize,filterSize), activation="relu", padding="same"))
        self.LayerNorm_1 = TimeDistributed(LayerNormalization())
        self.Dropout_1 = Dropout(0.2)


        self.T_CNN_2 = TimeDistributed(Conv2D(filter*2, (filterSize+1,filterSize+1), activation="relu", padding="same"))
        self.LayerNorm_2 = TimeDistributed(LayerNormalization())
        self.Dropout_2 = Dropout(0.2)


        self.T_CNN_3 = TimeDistributed(Conv2D(filter, (filterSize+1,filterSize+1), activation="relu", padding="same"))
        self.LayerNorm_3 = TimeDistributed(LayerNormalization())
        self.Dropout_3 = Dropout(0.2)

        self.T_CNN_4 = TimeDistributed(Conv2D(filter/2, (filterSize+1, filterSize+1), activation="relu", padding="same"))
        self.LayerNorm_4 = TimeDistributed(LayerNormalization())
        self.Dropout_4 = Dropout(0.2)

        self.T_CNN_5 = TimeDistributed(Conv2D(1, (filterSize,filterSize), activation="relu", padding="same"))


        self.residual_block_1 = ResBlock(filter, filterSize)
        self.residual_block_2 = ResBlock(filter/2, filterSize+1)
        self.residual_block_3 = ResBlock(filter/4, filterSize+2)
        self.residual_block_4 = ResBlock(1, filterSize)


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

        # x = self.residual_block_1(inputs)
        # x = self.Dropout_1(x)
        #
        # x = self.residual_block_2(x)
        # x = self.Dropout_2(x)
        #
        # x = self.residual_block_3(x)
        # x = self.Dropout_3(x)
        #
        # x = self.residual_block_4(x)
        # # x = self.Dropout_1(x)


        return x



