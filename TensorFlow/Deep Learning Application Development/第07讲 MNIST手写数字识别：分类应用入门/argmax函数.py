#!/usr/bin/env python
# coding: utf-8

# # argmax()用法
import tensorflow as tf
import numpy as np

arr1 = np.array([1,3,2,5,7,0])
arr2 = np.array([[1.0,2,3],[3,2,1],[4,7,2],[8,3,2]])
print("arr1=",arr1)
print("arr2=\n",arr2)

argmax_1 = tf.argmax(arr1)
argmax_20 = tf.argmax(arr2,0)  # 指定第二个参数为0,按第一维（行）的元素取值，即同列的每一行
argmax_21 = tf.argmax(arr2,1)  # 指定第二个参数为1,按第一维（列）的元素取值，即同列的每一列
argmax_22 = tf.argmax(arr2,-1) # 指定第二个参数为-1,则第最后维的元素取值

with tf.Session() as sess:
    print(argmax_1.eval())
    print(argmax_20.eval())
    print(argmax_21.eval())
    print(argmax_22.eval())





