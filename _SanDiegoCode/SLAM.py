import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, multiply
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint


class SLAM (tf.keras.layers.Layer):
    """Class implementation of Sigmoid Liquid Attention Matrix"""
    def __init__(self, relu):
        super(SLAM, self).__init__()
        if(relu == False):
            self.dense = Dense(10, activation="linear")
        else:
            self.dense = Dense(10, activation="relu")

    def call(self, inputs):
        x = self.dense(inputs)
        TenByTen = tf.linalg.matmul(x,x, transpose_a=True) #outer product of x
        SigmoidTenByTen = tf.math.sigmoid(TenByTen)
        return SigmoidTenByTen


