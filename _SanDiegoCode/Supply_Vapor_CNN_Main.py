import tensorflow as tf
from Supply_VAPOR_Model_Preprocess import model_preprocess_CNN, model_preprocess
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from SaveBestModel import SaveBestModel
from kerasncp import wirings
from kerasncp.tf import LTCCell
import seaborn as sns
from datetime import datetime

from VAPORModel import VAPOR_Model

inputLen = 12
train_x_pv, validation_x_pv, train_x_aux, validation_x_aux, train_y, validation_y = model_preprocess_CNN(inputLen,supplyTotal=True, showFig=False, normalize=True)
# print(train_x_pv)
SLAM_Dense_Sizes = [32]
ConvBlocks = [128]
filterSizes = [2,3]
for filterSize in filterSizes:
    for ConvBlock in ConvBlocks:
        for SLAM_Dense_Size in SLAM_Dense_Sizes:
            model = VAPOR_Model(SLAM_Dense=SLAM_Dense_Size, ConvBlock_Size=ConvBlock, tensorLen=int(inputLen/4), filterSize=3)
            opt = tf.keras.optimizers.Adam(learning_rate=0.0005, decay=1e-6) #0.0005

            model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=opt,
                                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
            model.fit([train_x_pv, train_x_aux], train_y, batch_size=1, epochs=20, validation_data=([validation_x_pv,validation_x_aux],validation_y))

            time = datetime.now().strftime("%m-%d-%H-%M-%S")

            NAME = f"VAPOR-SLAM-Norm-{SLAM_Dense_Size}-ConvBlock-{ConvBlock}-FilterSize-{filterSize}-SOFTMAX-InpLen-{inputLen}Time-{time}"

            tensorboard = TensorBoard(log_dir=f'SLAM_Logs_v1/{NAME}', histogram_freq=1,
                                              write_images=True)  # tensorboard --logdir=SupplyLogsv3

            model.summary()
            save_best_model = SaveBestModel()
            history = model.fit([train_x_pv, train_x_aux], train_y, batch_size=1, epochs=20, validation_data=([validation_x_pv,validation_x_aux],validation_y), callbacks=[tensorboard, save_best_model])
