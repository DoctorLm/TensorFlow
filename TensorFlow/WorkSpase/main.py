import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle
print("TensorFlow版本:", tf.__version__)

df = pd.read_csv("dataset/boston_data/boston.csv", header=0)
# 显示数据摘要描述信息
print(df.describe())
print(df)

df = df.values
print(df)

df = np.array(df)
print(df)

# 特征数据归一化
# 对特征数据[0到11]列做（0-1）归一化
for i in range(12):
    df[:,i] = df[:,i]/(df[:,i].max()-df[:,i].min())

# x_data 为前12列特征数据
x_data = df[:,:12]

# y_data 为最后1列标签数据
y_data = df[:,12]

print(x_data,"\n shape=", x_data.shape)
print(y_data,"\n shape=", y_data.shape)

x = tf.placeholder(tf.float32, [None,12], name = "X") # 12个特征数据（12列）
y = tf.placeholder(tf.float32, [None,1], name = "Y")  # 1个标签数据(1列)

with tf.name_scope("Model"):
    w = tf.Variable(tf.random_normal([12,1], stddev=0.01), name="W") # w初始化值为shape=(12,1)的随机数
    b = tf.Variable(1.0, name="b") # b初始化值为1.0
    def model(x, w, b): # w和x是矩阵相乘，用matmul,不能用mutiply或*
        return tf.matmul(x, w) + b
    pred = model(x, w, b) # 预测计算操作，前向计算节点

# 训练模型
# 设置训练参数（超参数）
train_epochs = 50 # 迭代轮次
learning_rate = 0.01 # 学习率

# 定义均方差损失函数
# 定义损失函数
with tf.name_scope("LossFunction"):
    loss_function = tf.reduce_mean(tf.square(y-pred)) #均方误差

# 创建优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

# 声明会话
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# 迭代训练
loss_list = [] # 用于保存loss值的列表
for epoch in range (train_epochs):
    loss_sum = 0.0
    for xs, ys in zip(x_data, y_data):
        xs = xs.reshape(1,12)
        ys = ys.reshape(1,1)
        _, loss = sess.run([optimizer, loss_function], feed_dict={x: xs, y: ys})
        loss_sum = loss_sum + loss
        
    # 打乱数据顺序
    xvalues, yvalues = shuffle(x_data, y_data)

    b0temp = b.eval(session=sess)
    w0temp = w.eval(session=sess)
    loss_average = loss_sum/len(y_data)
    
    loss_list.append(loss_average) # 每轮添加一次
    print("epoch=",epoch+1,"loss=",loss_average,"b=",b0temp,"w=",w0temp)

plt.plot(loss_list)
plt.show()
# 应用模型预测
n = np.random.randint(506)
print(n)
x_test = x_data[n]

x_test = x_test.reshape(1,12)
predict = sess.run(pred, feed_dict={x: x_test})
print("预测值:%f" % predict)
target = y_data[n]
print("标签值:%f" % target)
