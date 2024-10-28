import tensorflow as tf
import os
tf.config.set_visible_devices([], 'GPU')
print(tf.__version__)
print(tf.sysconfig.get_build_info()["cudnn_version"])
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
a = tf.constant(1.)
b = tf.constant(2.)
print(a+b)
print('GPU:', tf.test.is_gpu_available())
print(tf.config.list_physical_devices('GPU'))