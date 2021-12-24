import tensorflow as tf
from SLAM import SLAM
import numpy as np

class VAPOR_Model(tf.keras.Model):
    def __init__(self):
        super(VAPOR_Model, self).__init__()
        self.aux = SLAM(relu=False)
        # self.main = MainModel()

    def call(self, inputs):
        x_main, x_aux = inputs
        x_aux_np = np.array(x_aux)
        for aux in x_aux_np:
            print(aux)
            x = self.aux(aux)

        #return a 3x10x10 array
        return x