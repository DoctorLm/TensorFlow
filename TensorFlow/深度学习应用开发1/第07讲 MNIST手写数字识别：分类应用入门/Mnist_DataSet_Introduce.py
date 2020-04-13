#!/usr/bin/env python
# coding: utf-8

# # 下载并读取数据
import tensorflow as tf

#import tensorflow.examples.tutorials.mnist.input_data as input_data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
print("训练集 train 数量:", mnist.train.num_examples,
      ",验证集 validation 数量:", mnist.validation.num_examples, 
      ",测试集 test 数量:", mnist.test.num_examples)

# # 查看train Data
print("train images shape:", mnist.train.images.shape, 
      "labels shape:", mnist.train.labels.shape)

len(mnist.train.images[0])

mnist.train.images[0].shape

mnist.train.images[0]

mnist.train.images[0].reshape(28,28)

# # 显示图像
import matplotlib.pyplot as plt

def plot_image(image):
    plt.imshow(image.reshape(28,28),cmap='binary')
    plt.show()

plot_image(mnist.train.images[1000])

# # 进一步了解reshape()
import numpy as np
int_array = np.array([i for i in range(64)])
print(int_array)

int_array.reshape(8,8)

int_array.reshape(4,16)

plt.imshow(mnist.train.images[20000].reshape(14,56),cmap='binary')

plt.imshow(mnist.train.images[20000].reshape(7,112),cmap='binary')

# # 认识标签
mnist.train.labels[1]

import numpy as np
np.argmax(mnist.train.labels[1])
# # one hot 独热编码
mnist_no_one_hot = input_data.read_data_sets("MNIST_data", one_hot=False)
print(mnist_no_one_hot.train.labels[0:10])

# # 批量读取数据
print(mnist.train.images[0:10])

print(mnist.train.labels[0:10])

batch_images_xs, batch_labels_ys = mnist.train.next_batch(batch_size=10)

print(batch_images_xs.shape, batch_labels_ys.shape)

print(batch_images_xs)
print(batch_labels_ys)

batch_images_xs, batch_labels_ys = mnist.train.next_batch(batch_size=10)
print(batch_labels_ys)

# # 读取验证数据 validation data
print('validation images:', mnist.validation.images.shape,
      'labels:', mnist.validation.labels.shape)
# # 读取测试数据 test data
print('test images:', mnist.test.images.shape,
      'labels:', mnist.test.labels.shape)
# # 载入数据
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
# # 构建模型
# ## 定义x和y的占位符
# mnist 中每张图片共28*28=784个像素点
x = tf.placeholder(tf.float32, [None, 784], name="X")

# 0-9 一共10个数字=>10个类别
y = tf.placeholder(tf.float32, [None, 10], name="Y")

# # 创建变量
# 在本案例中，以正态分布的随机数初始化权重W，以常数0初始化偏置b
#定义变量
W = tf.Variable(tf.random_normal([784,10]),name="W")
b = tf.Variable(tf.zeros([10]), name="b")

# ### 在神经网络中，权值W的初始值通常设为正态分布的随机数，偏置项b的初始值通常也设为的随机数或常数。在Tensorflow中,通常利用以下函数实现正态分布随机数的生成：
# # 用单个神经元构建神经网络
forward = tf.matmul(x,W)+b # 前向计算

# # 关于Softmax Regression
#     当我们处理多分类任务时，通常需要使用Softmax Regression模型。
#     Softmax Regression会对每一类别估算出一个概率。
# # 工作原理：
#     将判定为某一类的特征相加，然后将这些特征转化为判定是这一类的概率。

pred = tf.nn.softmax(forward) #Sotfmax分类

# # 训练模型
# ## 设置训练参数
train_epochs = 50 # 训练轮数
batch_size = 100 # 单次训练样本数（批次大小）
total_batch = int(mnist.train.num_examples/batch_size) # 一轮训练有多少批次
display_step = 1 # 显示粒度
learning_rate = 0.01 # 学习率

# ## 定义损失函数
# 定义交叉熵损失函数
loss_function = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

# ## 选择优化器
# 梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

# ## 定义准确率
# 检查预测类别tf.argmax(pred,1)与实际类别tf.argmax(y,1)的匹配情况
correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))

# 准确率，将布乐值转化为浮点数，并计算平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session() # 声明会话
init = tf.global_variables_initializer() # 变量初始化
sess.run(init)

# # 模型训练
#开始训练
for epoch in range(train_epochs):
    for batch in range(total_batch):
        xs, ys = mnist.train.next_batch(batch_size) # 读取批次数据
        sess.run(optimizer,feed_dict = {x: xs, y: ys})  # 执行批次训练
    # total_batch个批次训练完成后,使用验证数据计算误差与准确率;验证集没有分批
    loss,acc = sess.run([loss_function, accuracy], feed_dict={x: mnist.validation.images, y: mnist.validation.labels})
    # 打印训练过程中的详细信息
    if(epoch+1) % display_step ==0:
        print("Train Epoch:%02d" % (epoch+1), "Loss={:.9f}".format(loss), "Accuracy={:.4f}".format(acc))
print("Train Finished!")

# ### 从上述打印结果可以看出损失值Loss是趋于更小的，同时，准确率Accuracy越来越高。
# # 评估模型
# ### 完成训练后，在 测试集 上评估模型的准确率
accu_test = sess.run(accuracy,feed_dict={x: mnist.test.images, y: mnist.test.labels})
print("Test Accuracy:", accu_test)

# ### 完成训练后,在 验证集 上评估模型的准确率
accu_validation = sess.run(accuracy, feed_dict={x: mnist.validation.images, y: mnist.validation.labels})
print("Test Accuracy:", accu_validation)

# ### 完成训练后,在 训练集 上评估模型的准确率
accu_train = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels})
print("Test Accuracy:", accu_train)

# # 进行预测
# ### 在建立模型并进行训练后，若认为准确率可以接受，则可以使用些模型进行预测
# 由于pred预测结果是one-hot编码格式，所以需要转换为0~9数字
prediction_result = sess.run(tf.argmax(pred,1), feed_dict={x: mnist.test.images})

# ### 查看预测结果
# 查看预测结果中的前10项
prediction_result[0:10]

# # 定义可视化函数
import matplotlib.pyplot as plt
import numpy as np
def plot_images_labels_prediction(images,     # 图像列表
                                  labels,     # 标签列表
                                  prediction, # 预测值列表
                                  index,      # 从第index个开始显示
                                  num=10):    # 缺省一次显示 10 幅
    fig = plt.gcf() # 获取当前图表，Get Current Figure
    fig.set_size_inches(10, 12) # 1英寸等于2.54cm
    if num > 25:
        num = 25 # 最多显示25个子图
    for i in range(0, num):
        ax = plt.subplot(5,5,i+1) # 获取当前要处理的子图
        ax.imshow(np.reshape(images[index], (28, 28)), cmap="binary") # 显示第index个图像

        title = "label=" + str(np.argmax(labels[index])) # 构建该图上要显示的title信息
        if len(prediction) > 0:
            title += ", predict=" + str(prediction[index])
        
        ax.set_title(title, fontsize=10) # 显示图上的title信息
        ax.set_xticks([]) # 不显示坐标轴
        ax.set_yticks([])
        index += 1
    plt.show()

plot_images_labels_prediction(mnist.test.images,
                              mnist.test.labels, 
                              prediction_result, 10, 25)

'''从上面结果可知，通过30次迭代所训练的由单个神经元构成的神经网络模型，在测试集上能够有百分之八十以上的准确率。接下来，我们将尝试加宽和加深模型，看看能否得到更高的准确率'''
# Tensorflow提供，带softmax的交叉熵函数
#loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=forward))

