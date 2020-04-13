#!/usr/bin/env python
# coding: utf-8
# 读取数据
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import scale
print("TensorFlow版本:", tf.__version__)

# 通过Pandas导入数据
# 读取数据文件
df = pd.read_csv("boston_data/boston.csv", header=0)
#显示数据摘要描述信息
print(df.describe())
df.head(3)
df.tail(3)

#获取df的值
ds = df.values
print(ds)
print(ds.shape)
print(ds)

x_data = ds[:,:12]
y_data = ds[:,12]

print('x_data shape=', x_data.shape)
print('y_data shape=', y_data.shape)

# for i in range(12):
#     x_data[:,i] = (x_data[:,i]-x_data[:,i].min())/(x_data[:,i].max()-x_data[:,i].min())

train_num = 300 #训练集的数目
valid_num = 100 #验证集的数目
test_num = len(x_data) - train_num - valid_num #测试集的数目 = 506 - 训练集的数目 - 验证集的数目
#训练集划分
x_train = x_data[:train_num]
y_train = y_data[:train_num]
#验证集划分
x_valid = x_data[train_num:train_num+valid_num]
y_valid = y_data[train_num:train_num+valid_num]
#测试集划分
x_test = x_data[train_num+valid_num:train_num+valid_num+test_num]
y_test = y_data[train_num+valid_num:train_num+valid_num+test_num]

x_train = tf.cast(scale(x_train), dtype=tf.float32)
x_valid = tf.cast(scale(x_valid), dtype=tf.float32)
x_test = tf.cast(scale(x_test), dtype=tf.float32)

def model(x, w, b):
    return tf.matmul(x, w) + b

W = tf.Variable(tf.random.normal([12, 1], mean=0.0, stddev=1.0, dtype=tf.float32))
B = tf.Variable(tf.zeros(1), dtype=tf.float32)

print(W)
print(B)

training_epochs = 50
learning_rate = 0.01
batch_size = 10

#采用均方差作为损失函数
def loss(x, y, w, b):
    err = model(x, w, b) - y
    squared_err = tf.square(err)
    return tf.reduce_mean(squared_err)

def grad(x, y, w, b):
    with tf.GradientTape() as tape:
        loss_ = loss(x, y, w, b)
    return tape.gradient(loss_, [w, b])

optimizer = tf.keras.optimizers.SGD(learning_rate)

loss_list_train =[]
loss_list_valid =[]
total_step = int(train_num/batch_size)

for epoch in range(training_epochs):
    for step in range(total_step):
        xs = x_train[step*batch_size:(step+1)*batch_size,:]
        ys = y_train[step*batch_size:(step+1)*batch_size]
        
        grads = grad(xs, ys, W, B)
        optimizer.apply_gradients(zip(grads, [W, B]))
    
    loss_train = loss(x_train, y_train, W, B).numpy()
    loss_valid = loss(x_valid, y_valid, W, B).numpy()
    loss_list_train.append(loss_train)
    loss_list_valid.append(loss_valid)
    print("epoch={:3d},train_loss={:.4f},valid_loss={:.4f}".format(epoch+1, loss_train, loss_valid))

# 可视化
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(loss_list_train, 'blue', label="Train Loss")
plt.plot(loss_list_valid, 'red', label="Valid Loss")
plt.legend(loc=1)

print("Test_loss:{:.4f}".format(loss(x_test, y_test, W, B).numpy()))

test_house_id = np.random.randint(0, test_num)
y = y_test[test_house_id]

y_pred = model(x_test, W, B)[test_house_id]
y_predit = tf.reshape(y_pred,()).numpy()

print("House id", test_house_id, "Actual value", y, "Predicted value", y_predit)
