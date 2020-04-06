#!/usr/bin/env python
# coding: utf-8

# # Tensorflow实现多变量线性回归
# # 1 数据导入和处理
# ## 导入相关库
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

# 数据集简介
# 本数据集包含与波士顿房价相关的金工个因素
# CRIM：城镇人均犯罪率
# ZN：住宅地超过25000 sq.ft.的比例
# INDUS：城镇非零售商用土地的比例
# CHAS：Charles河空变量（如果边界是河流，则为1;否则，为0）
# NOX：一氧化氮浓度
# RM：住宅平均房间数
# AGE：1940年之前建成的自用房屋比例
# DIS：到波士顿5个中心区域的加权距离
# RAD：辐射性公路的靠近指数]
# TAX：每1万美元的全值财产税率
# PTRATIO：城镇师生比例
# LSTAT：人口中地位低下者的比例
# MEDV：自住房的产均房价，单位：千美元
# 数据集以CSV格式存储，可通过Pandas库读取开进行格式转换
# Pandas库可以帮助我们快速读取常规大小的数据文件
# 能够读取CVS文件，文本文件、MS Excel、SQL数据库以及用于科学用途的HDF5格式文件
# 自动转为Numpy的多维阵列

# 通过Pandas导入数据
# 读取数据文件
df = pd.read_csv("/Users/lumin/Documents/GitHub.com/TensorFlow/TensorFlow/WorkSpase/dataset/boston_data/boston.csv", header=0)
# 显示数据摘要描述信息
print(df.describe())

# 显示所有信息
print(df)

# ## 载入本示例所需数据
# 获取df的值
df = df.values
print(df)

# 把df转换为np数组
df = np.array(df)
print(df)

# ## 特征数据归一化
# 对特征数据[0到11]列做（0-1）归一化
for i in range(12):
    df[:,i] = df[:,i]/(df[:,i].max()-df[:,i].min())

# x_data 为前12列特征数据
x_data = df[:,:12]

# y_data 为最后1列标签数据
y_data = df[:,12]

print(x_data,"\n shape=", x_data.shape)
print(y_data,"\n shape=", y_data.shape)

# 2 构建模型
# 定义特征数据和标签数据的占位符
# shape中None表示行的数量未知，在实际训练时决定一次代入多少行样本，从一个样本的随机SDG到批量SDG都可以
x = tf.placeholder(tf.float32, [None,12], name = "X") # 12个特征数据（12列）
y = tf.placeholder(tf.float32, [None,1], name = "Y")  # 1个标签数据(1列)

# 创建变量、定义模型
# 定义了一个命名空间
with tf.name_scope("Model"):
    
    # w初始化值为shape=(12,1)的随机数
    w = tf.Variable(tf.random_normal([12,1], stddev=0.01), name="W")
    
    # b初始化值为1.0
    b = tf.Variable(1.0, name="b")
    
    # w和x是矩阵相乘，用matmul,不能用mutiply或*
    def model(x, w, b):
        return tf.matmul(x, w) + b
    
    # 预测计算操作，前向计算节点
    pred = model(x, w, b)

# 3 训练模型
# 设置训练参数（超参数）
# 迭代轮次
train_epochs = 50

# 学习率
learning_rate = 0.01

# 定义均方差损失函数
# 定义损失函数
with tf.name_scope("LossFunction"):
    loss_function = tf.reduce_mean(tf.pow(y-pred,2)) #均方误差

# 选择优化器
# 常用优化器包括：
# tf.train.GradientDescentOptimizer
# tf.train.AdadeltaOptimizer
# tf.train.AdagradOptimizer
# tf.train.AdagradDAOptimizer
# tf.train.MomentumOptimizer
# tf.train.AdamOptimizer
# tf.train.FtrlOptimizer
# tf.train.ProximalGradientDescentOptimizer
# tf.train.ProximalAdagradOptimizer
# tf.train.RMSPropOptimizer

# 创建优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

# 声明会话
sess = tf.Session()
# 定义初始化变量的操作
init = tf.global_variables_initializer()

# 为TensorBoard可视化准备数据
# 设置日志存储目录
logdir='log'

# 创建一个操作，用于记录损失值loss，后面在TensorBoard中SCALARS栏可见
sum_loss_op = tf.summary.scalar("loss", loss_function)

# 把所有需要记录摘要日志文件合并，方便一次决性写入
merged = tf.summary.merge_all()

# 启动会话
sess.run(init)

# 创建摘要的文件写入器（FileWriter)
# 创建摘要writer, 将计算图写入摘要文件，后面在TensorBoard中GRAPHS栏可见
writer = tf.summary.FileWriter(logdir, sess.graph)

# 迭代训练
loss_list = [] # 用于保存loss值的列表
 
for epoch in range (train_epochs):
    loss_sum = 0.0
    for xs, ys in zip(x_data, y_data):
        xs = xs.reshape(1,12)
        ys = ys.reshape(1,1)
        _, summary_str, loss = sess.run([optimizer, sum_loss_op, loss_function], feed_dict={x: xs, y: ys})
        
        writer.add_summary(summary_str, epoch)
        loss_sum = loss_sum + loss
        
    # 打乱数据顺序
    x_data, y_data = shuffle(x_data, y_data)

    b0temp = b.eval(session=sess)
    w0temp = w.eval(session=sess)
    loss_average = loss_sum/len(y_data)
    
    loss_list.append(loss_average) # 每轮添加一次
    
    print("epoch=",epoch+1,"loss=",loss_average,"b=",b0temp,"w=",w0temp)

# 深度学习中对于网络的训练是参数更新的过程，需要注意一种情况就是输入数据未做归一化时，如果前向传播结果[0，0，0，1，0，0，0，0]这种形式，
# 而真实结果是[1，0，0，0，0，0，0，0]，此时由于得出的结论不惧有概率性，而是错误估计前向传播会使得权重和偏置值变的无穷大，导致数据溢出，也就出现了nan问题
# 可视化
plt.plot(loss_list)

# 应用模型预测
n = np.random.randint(506)
print(n)
x_test = x_data[n]

x_test = x_test.reshape(1,12)
predict = sess.run(pred, feed_dict={x: x_test})
print("预测值:%f" % predict)
target = y_data[n]
print("标签值:%f" % target)
