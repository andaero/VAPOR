import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, multiply, LSTM, Flatten
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint


class SLAM (tf.keras.layers.Layer):
    """Class implementation of Sigmoid Liquid Attention Matrix"""
    def __init__(self, dense, matrixSize, relu=False):
        super(SLAM, self).__init__()
        self.matrixSize = matrixSize
        if(relu == False):
            self.dense = Dense(dense, activation="linear")
            # self.dense_2 = Dense(dense, activation="linear")
            self.dense_3 = Dense(matrixSize, activation="linear")

            # self.LSTM = LSTM(dense, return_sequences=False)
            # self.Flatten = Flatten()
            # self.dense_2 = Dense(128, activation="linear")
            # self.dense_3 = Dense(tf.math.square(matrixSize), activation="linear")

        else:
            self.dense = Dense(dense, activation="relu")
            self.dense_2 = Dense(matrixSize, activation="linear")


    def call(self, inputs):

        #LINEAR PROJECTION VERSION
        x = self.dense(inputs)
        # x = self.dense_2(x)
        x = self.dense_3(x)

        #LSTM VERSION
        # x = self.LSTM(inputs)
        # x = self.Flatten(x)
        # x = self.dense_2(x)
        # x = self.dense_3(x)

        # print("X SHAPE", x.shape)
        # print("Transposed SHAPE", tf.transpose(x).shape)
        TenByTen = tf.linalg.matmul(x, x, transpose_a=True) #outer product of x

        # TenByTen = tf.reshape(x, [self.matrixSize, self.matrixSize])
        # print("TEN BY TEN SHAPE", TenByTen.shape)
        SigmoidTenByTen = tf.math.sigmoid(TenByTen)
        SoftMaxTenByTen = tf.math.softmax(TenByTen)
        return SoftMaxTenByTen


