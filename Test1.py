import tensorflow as tf
import numpy as np

sequential_data = np.empty([2, 2, 2])  # numpy array contains the sequences without accounting for the dimensionality
x = np.array([[2,2],[2,2]])
sequential_data[0] = x
sequential_data[1] = x


print(sequential_data)