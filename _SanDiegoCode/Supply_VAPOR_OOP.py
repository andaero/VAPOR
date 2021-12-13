import tensorflow as tf
from SupplyVapor import SupplyVapor
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



seq_lens = [36,48,24]

#CREATING THE MODEL

EPOCHS = 50
BatchSizes = [64]
learning_rs = [0.0005]
layers = [2,3]
dense = 2;
dBatchSize = 1;
dropout = 0.2;

inter = [80,64]
command_neurons = [64,64]
sensory_fanout = [24,20]
inter_fanout = [24,20]
motor_fanin = [24,16]
recurrent = [32,32]



for seq_len in seq_lens:
    for BatchSize in BatchSizes:
        for learning_r in learning_rs:
            for layer in layers:
                for i in range(len(inter)): #range is EXCLUSIVE
                    train_x, train_y, validation_x, validation_y = model_preprocess(seq_len=seq_len)

                    # LIQUID FIRST TEST
                    Vapor = SupplyVapor(train_x, train_y, validation_x, validation_y, seq_len, EPOCHS)
                    Vapor.rnn_cell_init(inter[i], command_neurons[i], sensory_fanout[i], inter_fanout[i], motor_fanin[i],
                                        recurrent[i])
                    Vapor.Liquid_LSTM_init(layer, BatchSize, dropout, learning_r)

                    # LSTM FIRST TEST
                    # Vapor = SupplyVapor(train_x,train_y, validation_x, validation_y, seq_len,EPOCHS)
                    # Vapor.rnn_cell_init(inter[i], command_neurons[i], sensory_fanout[i], inter_fanout[i], motor_fanin[i], recurrent[i])
                    # Vapor.LSTM_Liquid_init(layer,BatchSize,dropout,learning_r)

