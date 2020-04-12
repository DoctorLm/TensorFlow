#coding:utf-8
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
print("TensorFlow版本:", tf.__version__)
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("/Users/lumin/Documents/dataset/MNIST_data/", one_hot=True)

print("训练集      train 数量:", mnist.train.num_examples,
    "\n验证集 validation 数量:", mnist.validation.num_examples,
    "\n测试集       test 数量:", mnist.test.num_examples)

print("train images shape:", mnist.train.images.shape,
    "\ntrain labels shape:", mnist.train.labels.shape)


