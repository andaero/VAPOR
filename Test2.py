import keras.layers
import matplotlib.pyplot as plt
from kerasncp import wirings
from kerasncp.tf import LTCCell
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, MaxPool2D, MaxPool3D, Flatten, RNN, InputLayer
import seaborn as sns
import numpy as np

N = 48 # Length of the time-series
# Input feature is a sine and a cosine wave
data_x = np.stack(
    [np.sin(np.linspace(0, 3 * np.pi, N)), np.cos(np.linspace(0, 3 * np.pi, N))], axis=1
)
data_x = np.expand_dims(data_x, axis=0).astype(np.float32)  # Add batch dimension
# Target output is a sine with double the frequency of the input signal
data_y = np.sin(np.linspace(0, 6 * np.pi, N)).reshape([1, N, 1]).astype(np.float32)
print("data_x.shape: ", str(data_x.shape))
print("data_y.shape: ", str(data_y.shape))

# Let's visualize the training data
sns.set()
plt.figure(figsize=(6, 4))
plt.plot(data_x[0, :, 0], label="Input feature 1")
plt.plot(data_x[0, :, 1], label="Input feature 1")
plt.plot(data_y[0, :, 0], label="Target output")
plt.ylim((-1, 1))
plt.title("Training data")
plt.legend(loc="upper right")
plt.show()

fc_wiring = wirings.FullyConnected(8,1)
ltc_cell = LTCCell(fc_wiring)

#Creating LTC model
model = Sequential()
model.add(InputLayer(input_shape=(None,2)))
model.add(RNN(ltc_cell,return_sequences=True))
opt = Adam(learning_rate=0.01, decay=1e-6)

# model.add(Dense(64,activation="relu"))
# model.add(Dense(2,activation="linear"))

model.compile(optimizer=opt,loss="mean_squared_error")
model.summary()

#Plotting wiring
sns.set_style("white")
plt.figure(figsize=(6,4))
legend_handles = ltc_cell.draw_graph(draw_labels=True)
plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1,1))
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.show()

#Prediction of LTC b4 training
sns.set()
prediction = model(data_x).numpy()
plt.figure(figsize=(6,4))
plt.plot(data_y[0,:,0], label="Target output")
plt.plot(prediction[0,:,0], label="LTC output")
plt.ylim((-1,1))
plt.title("Before training")
plt.legend(loc="upper right")
plt.show()
hist = model.fit(x=data_x,y=data_y,batch_size=1,epochs=300,verbose=1)

sns.set()
plt.figure(figsize=(6,4))
plt.plot(hist.history["loss"], label="Training loss")
plt.legend(loc="upper right")
plt.xlabel("Training steps")
plt.show()

# How does the trained model now fit to the sinusoidal function?
prediction = model(data_x).numpy()
plt.figure(figsize=(6, 4))
plt.plot(data_y[0, :, 0], label="Target output")
plt.plot(prediction[0, :, 0], label="LTC output",linestyle="dashed")
plt.ylim((-1, 1))
plt.legend(loc="upper right")
plt.title("After training")
plt.show()


rnd_wiring = wirings.Random(8,1,sparsity_level=0.75)
sparse_cell = LTCCell(rnd_wiring)

sparse_model = Sequential()
sparse_model.add(InputLayer(input_shape=(None,2)))
sparse_model.add(RNN(sparse_cell,return_sequences=True))
sparse_model.compile(optimizer=opt,loss="mean_squared_error")

# Plot the wiring
sns.set_style("white")
plt.figure(figsize=(6, 4))
legend_handles = sparse_cell.draw_graph(draw_labels=True)
plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.show()

#Training random sparse model
hist_rand = sparse_model.fit(x=data_x, y=data_y, batch_size=1, epochs=400,verbose=1)
sns.set()
plt.figure(figsize=(6, 4))
plt.plot(hist.history["loss"], label="Fully-connected")
plt.plot(hist_rand.history["loss"], label="Random (75% sparsity)")
plt.legend(loc="upper right")
plt.xlabel("Training steps")
plt.ylabel("Training loss")
plt.show()

#NCP WIRING
ncp_arch = wirings.NCP(
    inter_neurons=3,  # Number of inter neurons
    command_neurons=4,  # Number of command neurons
    motor_neurons=1,  # Number of motor neurons
    sensory_fanout=2,  # How many outgoing synapses has each sensory neuron
    inter_fanout=2,  # How many outgoing synapses has each inter neuron
    recurrent_command_synapses=3,  # Now many recurrent synapses are in the
    # command neuron layer
    motor_fanin=4,  # How many incomming syanpses has each motor neuron
)
ncp_cell = LTCCell(ncp_arch)

ncp_model = keras.models.Sequential(
    [
        keras.layers.InputLayer(input_shape=(None, 2)),
        keras.layers.RNN(ncp_cell, return_sequences=True),
    ]
)
ncp_model.compile(
    optimizer=opt, loss='mean_squared_error'
)
sns.set_style("white")
plt.figure(figsize=(6, 4))
legend_handles = ncp_cell.draw_graph(draw_labels=True, neuron_colors={"command": "tab:cyan"})
plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1.1, 1.1))
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.show()

hist_ncp = ncp_model.fit(x=data_x, y=data_y, batch_size=1, epochs=400,verbose=1)
# This may take a while (training the LTC model)
sns.set()
plt.figure(figsize=(6, 4))
plt.plot(hist.history["loss"], label="Fully-connected")
plt.plot(hist_rand.history["loss"], label="Random (75% sparsity)")
plt.plot(hist_ncp.history["loss"], label="NCP")
plt.legend(loc="upper right")
plt.xlabel("Training steps")
plt.ylabel("Training loss")
plt.show()