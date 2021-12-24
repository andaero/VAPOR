import tensorflow as tf
from Supply_VAPOR_Model_Preprocess import model_preprocess_CNN, model_preprocess
from VAPORModel import VAPOR_Model as VAPOR


train_x_pv, validation_x_pv, train_x_aux, validation_x_aux, train_y, validation_y = model_preprocess_CNN(12,supplyTotal=False, showFig=False)

model = VAPOR()
opt = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)

model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=opt,
                      metrics=[tf.keras.metrics.RootMeanSquaredError()])
model.fit([train_x_pv, train_x_aux], train_y, batch_size=64, epochs=20)
#validation_data=([validation_x_pv, validation_x_aux], validation_y))
