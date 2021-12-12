
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
from SupplyVapor import SupplyVapor
from Supply_LSTM_Solar_Only import weatherPreprocessing, scaleData
from Supply_VAPOR_Model_Preprocess import model_preprocess


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

drop_remainder = True



seq_len = 24
train_x, train_y, validation_x, validation_y = model_preprocess(seq_len=seq_len)

#CREATING THE MODEL

EPOCHS = 80
BatchSizes = [64,32]
learning_rs = [0.0005]
layers = [2]
dense = 2;
dBatchSize = 1;
dropout = 0.2;

inter = [32,20]
command_neurons = [32,20]
sensory_fanout = [16,10]
inter_fanout = [16,10]
motor_fanin = [16,10]
recurrent = [20,18]




for BatchSize in BatchSizes:
    for learning_r in learning_rs:
        for layer in layers:
            for i in range(len(inter)):
                Vapor = SupplyVapor(train_x,train_y, validation_x, validation_y, seq_len)
                Vapor.rnn_cell_init(inter[i], command_neurons[i], sensory_fanout[i], inter_fanout[i], motor_fanin[i], recurrent[i])
                Vapor.LSTM_Liquid_init(layer,BatchSize,dropout,learning_r)
                Vapor.train_model()
