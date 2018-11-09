import tensorflow as tf
import numpy as np


#create a np array
tensor_1d = np.array([1.45, -1, 0.2, 102.1])
print(tensor_1d)
print(tensor_1d.dtype)

#convert_to_tensor will convert a np arry to a tf tensor
tensor = tf.convert_to_tensor(tensor_1d, dtype=tf.float64)

with tf.Session() as session:
	print(session.run(tensor))
	print(session.run(tensor[0]))
	print(session.run(tensor[1]))