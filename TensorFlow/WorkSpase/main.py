#coding:utf-8
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
#允许使用TF2执行模式
#import tensorflow.compat.v2 as tf
#tf.enable_v2_behavior()
# 禁用TF2执行模式
#import tensorflow.v1 as tf
#tf.disable_eager_execution()
import tensorflow as tf
print("TensorFlow版本:", tf.__version__)
