import tensorflow as tf
import numpy as np



#Quick method to convert between np arrays and tf tensors
def convert(v, t=tf.float32):
	return tf.convert_to_tensor(v, dtype=t)

m1 = convert(np.array(np.random.rand(4, 4), dtype='float32'))
m2 = convert(np.array(np.random.rand(4, 4), dtype='float32'))
m3 = convert(np.array(np.random.rand(4, 4), dtype='float32'))
m4 = convert(np.array(np.random.rand(4, 4), dtype='float32'))
m5 = convert(np.array(np.random.rand(4, 4), dtype='float32'))

m_transpose = tf.transpose(m1)
m_mul = tf.matmul(m1, m2)
m_det = tf.matrix_determinant(m3)
m_inv = tf.matrix_inverse()