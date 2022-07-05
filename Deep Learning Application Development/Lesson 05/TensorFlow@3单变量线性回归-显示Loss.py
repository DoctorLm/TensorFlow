#!/usr/bin/env python
# coding: utf-8
# Tensorflow实现单变量线性回归
# 假设我们要学习的函数为线性函数y = 2x +1
# 生成数据
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# 设置随机数种子
np.random.seed(5)
# 直接采用np生成等差数列的方法，生成100个点，每个点的取值在-1~1之间
x_data = np.linspace(-1, 1, 100)
# y = 2x +1 + 噪声， 其中， 噪声的唯度与x_data一致
y_data = 2 * x_data + 1.0 + np.random.randn(*x_data.shape) * 0.4
# 画出随机生成数据的散点图
plt.scatter(x_data, y_data)
# 画出通过学习得到的目标线性函数 y = 2x + 1
plt.plot(x_data, 2 * x_data + 1.0, color='red', linewidth=3)
plt.show()

# 构建模型
# 定义x和y的占位符
# 定义训练数据的占位符，x是特征，y是标签值
x = tf.placeholder("float",name="x")
y = tf.placeholder("float",name="y")

# 构建回归模型
def model(x,w,b):
    return tf.multiply(x,w)+b

# 创建变量
# Tensorflow变量的声明函数是tf.Variable
# tf.Variable的作用是保存和更新参数
# 变量的初始值可以是随机数、常数，或是通过其他变量的初始值计算得到
# 构建线性函数的斜率，变量w
w = tf.Variable(1.0,name="w0")
# 构建线性函数的截距，变量b
b = tf.Variable(0.0,name="b0")

# pred是预测值，前向计算
pred = model(x,w,b)

# 训练模型
# 设置训练参数
# 迭代次数（训练轮数）
train_epochs = 10
# 学习率
learning_rate = 0.05
# 控制显示loss值的粒度
display_step = 10

# 关于学习率（learning_rate）的设置
# 学习率的作用：控制参数更新的幅度。
# 如果学习率设置过大，可能导致参数在极值附过来回摇摆，无法保证收敛。
# 如果学习率设置过小，虽然能保证收敛，但优化速度会大大降低，我们需要更多迭代次数才能达到较理想的优化效果
# 定义损失函数
# 损失函数用于描述预测值与真实值之间的误差，从而指导模型收敛方向
# 常见损失函数：均方差（Mean Square Error，MSE）和交叉熵（cross-entropy）
# 采用均方差作为损失函数
loss_function = tf.reduce_mean(tf.square(y-pred))

# 选择优化器
# 梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

# 声明会话
sess = tf.Session()

# 变量初始化
# 在真正执行计算之前，需将所有变量初始化
# 通过tf.global_variables_initializer 函数可实现对所有变量的初始化
init = tf.global_variables_initializer()
sess.run(init)

# 执行训练
# 开始训练，轮数为epoch，采用SGD随机梯度下降优化方法
step = 0 # 记录步数
loss_list = [] #用于保存loss值的列表

for epoch in range(train_epochs):
    for xs,ys in zip(x_data,y_data):
        _,loss=sess.run([optimizer,loss_function],feed_dict={x:xs,y:ys})
        
        # 显示损失值loss
        # display_step:控制报告的粒度
        # 例如，如果display_step 设为 2 ,则将每训练2个样本输出一次损失值
        # 与超参数不同，修改display_step 不会更改模型所学习的规律
        loss_list.append(loss)
        step=step+1
        if step % display_step == 0:
            print("Train Epoch:%02d"%(epoch+1), "step:%03d"%(step), "loss={:.9f}".format(loss))
            
    b0temp=b.eval(session=sess)
    w0temp=w.eval(session=sess)
    plt.plot(x_data, w0temp * x_data + b0temp) #画图

# 从上图可以看出，由于本案例所拟合的模型较简单，训练3次这后已经接近收敛。
# 对于复杂模型，需要更多次训练才能收敛。
plt.plot(loss_list)
plt.show()
plt.plot(loss_list,'g2')
plt.show()

[x for x in loss_list if x>1]

# 打印结果
print("w:",sess.run(w)) # W的值应该在2附过
print("b:",sess.run(b)) # b的值应该在1附过

# 可视化
plt.scatter(x_data,y_data,label='Original data')
plt.plot(x_data,x_data*sess.run(w) + sess.run(b),label='Fitted line',color='r',linewidth=3)
plt.legend(loc=2) # 通过参数loc指定图例位置
plt.show()

# 进行预测
x_test = 3.21
predict = sess.run(pred, feed_dict={x: x_test})
print("预测值:%f" % predict)
target = 2 * x_test + 1.0
print("目标值:%f" % target)

predict = sess.run(w) * x_test + sess.run(b)
print("预测值:%f" % predict)
target = 2 * x_test + 1.0
print("目标值:%f" % target)

'''
以上是利用Tensorflow训练一个线性模型并进行预测的完整过程。
通过逐渐降低损失值loss来训练参数w和b拟合 y = 2x + 1 中的系数2和1。小结
通过一个简单的例子介绍了利用Tensorflow实现机器学习的思路，重点进解了下述步骤：
（1）生成人工数据集及其可视化
（2）构建线性模型
（3）定义损失函数
（4）定义工优化器、最小化损失函数
（5）训练结果的可视化
（6）利用学习到的模型进行预测
'''
