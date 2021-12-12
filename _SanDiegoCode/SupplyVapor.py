from collections import deque
import glob
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
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
from Supply_LSTM_Solar_Only import SaveBestModel


class SupplyVapor:
    """Class implementation of VAPOR supply model"""

    def __init__(self, train_x, train_y, validation_x, validation_y, seq_len):
        self.train_x = train_x
        self.train_y = train_y
        self.validation_x = validation_x
        self.validation_y = validation_y

        self.model = Sequential()
        self.seq_len = seq_len

    def rnn_cell(self, inter, command_neurons, sensory_fanout, inter_fanout, motor_fanin, recurrent):
        self.inter = inter
        self.command_neurons = command_neurons
        self.sensory_fanout = sensory_fanout
        self.inter_fanout = inter_fanout
        self.motor_fanin = motor_fanin
        self.recurrent = recurrent
        wiring = wirings.NCP(
            inter_neurons=inter,  # Number of inter neurons
            command_neurons=command_neurons,  # Number of command neurons
            motor_neurons=1,  # Number of motor neurons
            sensory_fanout=sensory_fanout,  # How many outgoing synapses has each sensory neuron
            inter_fanout=inter_fanout,  # How many outgoing synapses has each inter neuron
            recurrent_command_synapses=recurrent,  # Now many recurrent synapses are in the
            # command neuron  layer
            motor_fanin=motor_fanin,  # How many incoming syanpses has each motor neuron
        )
        self.rnn_cell = LTCCell(wiring)

    def Liquid_LSTM_init(self,layers,BatchSize, dropout, learning_r):
        self.learning_r = learning_r

        time = datetime.now().strftime("%m-%d-%H-%M-%S")
        self.NAME = f"Liquid-LSTM-SeqLen-{self.seq_len}- Dropout-{dropout}-Cmnd N-{self.command_neurons}-Intr N-{self.inter}-Snsry F-{self.sensory_fanout}-Intr F-{self.inter_fanout}-Motor Fanin-{self.motor_fanin}-Recurrent-{self.recurrent}-Filters-{BatchSize}-Layers{layers}-Learning R-{self.learning_r}-Time-{time}"

        """Liquid NN"""
        self.model.add(InputLayer(input_shape=self.train_x.shape[1:]))
        try:
            self.model.add(RNN(self.rnn_cell, return_sequences=True))
        except:
            print("Make sure to init rnn_cell first!")

        """LSTM"""
        for x in range(layers-1):
            self.model.add(LSTM(BatchSize, input_shape=(self.train_x.shape[1:]), return_sequences=True))
            self.model.add(Dropout(dropout))
            self.model.add(BatchNormalization())
        self.model.add(LSTM(BatchSize, input_shape=(self.train_x.shape[1:]), return_sequences=False))
        self.model.add(Dropout(dropout))
        self.model.add(BatchNormalization())

        """ANN"""
        self.model.add(Flatten())
        self.model.add(Dense(BatchSize, activation="relu"))
        self.model.add(Dense(1, activation="linear"))

        """Model Training"""
        self.train_model()

    def LSTM_Liquid_init(self,layers,BatchSize,dropout,learning_r):
        self.learning_r = learning_r

        time = datetime.now().strftime("%m-%d-%H-%M-%S")
        self.NAME = f"Liquid-LSTM-SeqLen-{self.seq_len}- Dropout-{dropout}-Cmnd N-{self.command_neurons}-Intr N-{self.inter}-Snsry F-{self.sensory_fanout}-Intr F-{self.inter_fanout}-Motor Fanin-{self.motor_fanin}-Recurrent-{self.recurrent}-Filters-{BatchSize}-Layers{layers}-Learning R-{self.learning_r}-Time-{time}"

        """LSTM"""
        for x in range(layers-1):
            self.model.add(LSTM(BatchSize, input_shape=(self.train_x.shape[1:]), return_sequences=True))
            self.model.add(Dropout(dropout))
            self.model.add(BatchNormalization())
        self.model.add(LSTM(BatchSize, input_shape=(self.train_x.shape[1:]), return_sequences=True))
        self.model.add(Dropout(dropout))
        self.model.add(BatchNormalization())

        """Liquid NN"""
        try:
            self.model.add(RNN(self.rnn_cell, input_shape=(self.train_x.shape[1:]),return_sequences=True))
        except:
            print("Make sure to init rnn_cell first!")

        """ANN"""
        self.model.add(Flatten())
        self.model.add(Dense(BatchSize, activation="relu"))
        self.model.add(Dense(1, activation="linear"))


        """Model Training"""
        self.train_model()

    def train_model(self):
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_r, decay=1e-6)
        self.model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=opt,
                      metrics=[tf.keras.metrics.RootMeanSquaredError()])  # tf.keras.metrics.RootMeanSquaredError()

        self.model.summary()

        # plot_wiring()
        tensorboard = TensorBoard(log_dir=f'SupplyLogsv3/{self.NAME}', histogram_freq=1,
                                  write_images=True)  # tensorboard --logdir=SupplyLogsv3

        # filepath = "eNet-{epoch:02d}-{mean_absolute_percentage_error:.3f}"
        # checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor=['mean_absolute_percentage_error'] , verbose=1, save_best_only=True, mode='max'))

        save_best_model = SaveBestModel()
        history = self.model.fit(self.train_x, self.train_y, batch_size=self.BatchSize, epochs=self.EPOCHS,
                            validation_data=(self.validation_x, self.validation_y), callbacks=[tensorboard, save_best_model])

        score = self.model.evaluate(self.validation_x, self.validation_y, verbose=0)
        print('Test loss:', score[0])
        print('Test RMSE:', score[1])
        for s in score:
            print(s)



