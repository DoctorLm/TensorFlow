#coding:utf-8
import matplotlib.pyplot as plt
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

#定义输入和参数
x = tf.constant([[0.7, 0.5]])
w1= tf.Variable(tf.random.normal([2, 3], stddev=1, seed=1))
w2= tf.Variable(tf.random.normal([3, 1], stddev=1, seed=1))

#定义前向传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

with tf.compat.v1.Session() as sess:
    init_op = tf.compat.v1.global_variables_initializer()
    sess.run(init_op)
    
    print(sess.run(y))
