import tensorflow as tf
from Supply_VAPOR_Model_Preprocess import model_preprocess_CNN, model_preprocess

model_preprocess_CNN(12,supplyTotal=False)


# model = VAPORModel()
# model([x_main, x_aux])
#
# class VAPORModel(tf.keras.Model):
#
#     def __init__(self, name):
#         self.name = name
#         self.main = MainModel()
#         self.aux = SLAM()
#
#     def __call__(self, xs):
#         x_main, x_aux = xs
#         x = self.aux(x_aux)
#         ...
#         x = self.main(x)
#         return x