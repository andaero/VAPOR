import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, MaxPool2D, MaxPool3D, Flatten, RNN, Bidirectional, InputLayer
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras import mixed_precision
from functools import reduce

from scipy import stats
from kerasncp import wirings
from kerasncp.tf import LTCCell
import seaborn as sns
from datetime import datetime
from SaveBestModel import SaveBestModel

class SLAM:
    """Class implementation of Sigmoid Liquid Attention Matrix"""
    def __init__(self, inputVector, ):

