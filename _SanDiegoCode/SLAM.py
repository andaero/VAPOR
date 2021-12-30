import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, multiply
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint


class SLAM (tf.keras.layers.Layer):
    """Class implementation of Sigmoid Liquid Attention Matrix"""
    def __init__(self, dense, relu=False):
        super(SLAM, self).__init__()
        if(relu == False):
            self.dense = Dense(dense, activation="linear")
            self.dense_2 = Dense(10, activation="linear")

        else:
            self.dense = Dense(10, activation="relu")

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.dense_2(x)

        # print(x.shape)
        TenByTen = tf.linalg.matmul(x,x, transpose_a=True) #outer product of x
        # print("TEN BY TEN SHAPE", TenByTen.shape)
        SigmoidTenByTen = tf.math.sigmoid(TenByTen)
        SoftMaxTenByTen = tf.math.softmax(TenByTen)
        return SoftMaxTenByTen


