import tensorflow as tf
from SLAM import SLAM
import numpy as np

class VAPOR_Model(tf.keras.Model):
    def __init__(self):
        super(VAPOR_Model, self).__init__()
        self.aux = SLAM(relu=False)
        # self.main = MainModel()

    def call(self, inputs):
        # watch out for batch norm - weird w training=True
        x_main, x_aux = inputs

        aux_slices = tf.unstack(x_aux, axis=1) #axis=1 because 0th dimension is unknown
        main_slices = tf.unstack(x_main, axis=1) #axis=1 because 0th dimension is unknown

        print("AUX SHAPE" , aux_slices)
        print("MAIN SHAPE" , main_slices)


        # c = lambda i : i<len(aux_slices)
        # x = tf.while_loop(c, self.aux(), aux_slices)
        i = 0
        ta_aux = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)
        ta = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)

        for aux_slice in aux_slices:
            main_slice_squeezed= tf.squeeze(main_slices[i])

            print("MAIN SLICE SHAPE",main_slice_squeezed.shape)

            ta = ta.write(i, tf.linalg.matmul(main_slice_squeezed,self.aux(aux_slice)))
            i+=1
            print(i)

        ta_stacked= ta.stack() # Don't need to stack can just directly multiply by the 10x10 matrices
        # ta = tf.print(ta, [ta])

        #return a 3x10x10 array
        print(ta_stacked)
        return ta_stacked