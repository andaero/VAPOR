import tensorflow as tf
from SLAM import SLAM
import numpy as np

class SLAM_Layer(tf.keras.layers.Layer):
    def __init__(self, dense, tensorLen, relu=False):
        super(SLAM_Layer, self).__init__()
        self.aux = SLAM(dense, relu=relu)
        self.tensorLen = tensorLen

    def call(self, x_main, x_aux):
        # watch out for batch norm - weird w training=True

        aux_slices = tf.unstack(x_aux, axis=1) #axis=1 because 0th dimension is unknown
        main_slices = tf.unstack(x_main, axis=1) #axis=1 because 0th dimension is unknown

        # print("AUX SHAPE" , aux_slices)
        # print("MAIN SHAPE" , main_slices)


        # c = lambda i : i<len(aux_slices)
        # x = tf.while_loop(c, self.aux(), aux_slices)
        i = 0
        ta = tf.TensorArray(dtype=tf.float32, size=self.tensorLen, dynamic_size=False)

        for aux_slice in aux_slices:
            main_slice_squeezed= tf.squeeze(main_slices[i])

            outer_product = tf.linalg.matmul(main_slice_squeezed,self.aux(aux_slice))
            # print("MAIN SLICE SHAPE",main_slice_squeezed.shape)
            # print("AUX SLICE SHAPE",self.aux(aux_slice).shape)

            # print("OUTER PRODUCT: ", outer_product)
            ta = ta.write(i, outer_product)
            # ta = ta.write(i, main_slice_squeezed) #TESTING PURPOSES
            i += 1
            # print(i)
        # print("TA:", ta)
        ta_stacked= ta.stack() # Don't need to stack can just directly multiply by the 10x10 matrices
        # ta = tf.print(ta, [ta])
        # print(ta_stacked)
        ta_stacked_CNN_input = tf.expand_dims(ta_stacked, axis=-1)
        ta_stacked_CNN_input_2 = tf.expand_dims(ta_stacked_CNN_input, axis=0)

        #return a 3x10x10 array
        # print(ta_stacked_CNN_input.shape.as_list())
        return ta_stacked_CNN_input_2