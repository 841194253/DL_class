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

if tf.test.is_gpu_available():
    print("GPU is available")
else:
    print("GPU is not available, using CPU")

print(tf.test.gpu_device_name())
print(tf.config.experimental.set_visible_devices)
print('GPU:', tf.config.list_physical_devices('GPU'))
print('CPU:', tf.config.list_physical_devices(device_type='CPU'))
print(tf.config.list_physical_devices('GPU'))
print(tf.test.is_gpu_available())
# 输出可用的GPU数量
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# 查询GPU设备
