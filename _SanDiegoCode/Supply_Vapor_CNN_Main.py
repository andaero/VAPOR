import tensorflow as tf
from Supply_VAPOR_Model_Preprocess import model_preprocess_CNN, model_preprocess, model_preprocess_CNN_Twenty
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from SaveBestModel import SaveBestModel
from kerasncp import wirings
from kerasncp.tf import LTCCell
import seaborn as sns
from datetime import datetime

from VAPORModel import VAPOR_Model

inputLens = [168] #72 is good 96, 12, 168, 72,12
# print(train_x_pv)
SLAM_Dense_Sizes = [32]
ConvBlocks = [128]
filterSizes = [2]
CNN_Dense_Sizes = [64]
CNN_2_Units = [128]
Conv2Bools = [False]
for inputLen in inputLens:
    for filterSize in filterSizes:
        for CNN_Dense_Size in CNN_Dense_Sizes:
            for CNN_2_Unit in CNN_2_Units:
                for ConvBlock in ConvBlocks:
                    for Conv2Bool in Conv2Bools:
                        for SLAM_Dense_Size in SLAM_Dense_Sizes:

                            #INCLUDED HOUR VALUE
                            train_x_pv, validation_x_pv, train_x_aux, validation_x_aux, train_y, validation_y = model_preprocess_CNN(
                                inputLen, supplyTotal=True, showFig=False, normalize=True)
                            model = VAPOR_Model(SLAM_Dense=SLAM_Dense_Size, ConvBlock_Size=ConvBlock, filterSize=filterSize, CNN_Dense=CNN_Dense_Size, CNN_2=CNN_2_Unit, tensorLen=int(inputLen/4), Conv2Bool=Conv2Bool)
                            opt = tf.keras.optimizers.Adam(learning_rate=0.0005, decay=1e-6) #0.0005

                            model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=opt,
                                                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

                            time = datetime.now().strftime("%m-%d-%H-%M-%S")

                            NAME = f"VAPOR-SLAM No Layer Norm 10x10-{SLAM_Dense_Size}-Valid Padding-ConvBlock-{ConvBlock}-FilterSize-{filterSize}-ConvBlock_2-{Conv2Bool}-{CNN_2_Unit}" \
                                   f"-CNN Dense-{CNN_Dense_Size}-InpLen-{inputLen}-Time-{time}"
                            # NAME = f"No attention benchmark CNN"

                            print(NAME)
                            tensorboard = TensorBoard(log_dir=f'SLAM_Logs_v2/{NAME}', histogram_freq=1,
                                                              write_images=True)  # tensorboard --logdir=SupplyLogsv3

                            save_best_model = SaveBestModel()
                            history = model.fit([train_x_pv, train_x_aux], train_y, batch_size=1, epochs=20, validation_data=([validation_x_pv,validation_x_aux],validation_y), callbacks=[tensorboard, save_best_model])
