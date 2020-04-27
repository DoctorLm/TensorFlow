#coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
print("TensorFlow版本:", tf.__version__)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/Users/lumin/Documents/dataset/MNIST_data", one_hot=True)
print("训练集 train.num_examples      数量:", mnist.train.num_examples,
    "\n验证集 validation.num_examples 数量:", mnist.validation.num_examples,
    "\n测试集 test.num_examples       数量:", mnist.test.num_examples)
    
#定义全连接层函数
def fcn_layer(inputs,           #输入数据
              input_dim,        #输入神经元数量
              output_dim,       #输出神经元数量
              activation=None): #激活函数
    W = tf.Variable(tf.random.truncated_normal([input_dim, output_dim], stddev=0.1)) #以截断正态分布的随机数初始化W
    b = tf.Variable(tf.zeros([output_dim])) #以0初始化b
    XWb = tf.matmul(inputs, W) + b #建立表达式: inputs * W + b
    if activation is None: #默认有使用激活函数
        outputs = XWb
    else: # 若传入激活函数，则用其对输出结果进行变换
        outputs = activation(XWb)
    return outputs
    
# mnist 中每张图片共28*28=784个像素点
x = tf.compat.v1.placeholder(tf.float32, [None, 784], name="X")
# 0-9 一共10个数字=>10个类别
y = tf.compat.v1.placeholder(tf.float32, [None, 10], name="Y")

# 隐藏层神经元数量
H1_NN = 256 # 第1隐藏层神经元为256
H2_NN = 64  # 第2隐藏层神经元为64
H3_NN = 32  # 第3隐藏层神经元为32

# 构建隐藏层
h1 = fcn_layer(inputs=x,
               input_dim=784,
               output_dim=H1_NN,
               activation=tf.nn.relu)

h2 = fcn_layer(inputs=h1,
               input_dim=H1_NN,
               output_dim=H2_NN,
               activation=tf.nn.relu)

h3 = fcn_layer(inputs=h2,
               input_dim=H2_NN,
               output_dim=H3_NN,
               activation=tf.nn.relu)

# 构建输出层
forward = fcn_layer(inputs=h3,
               input_dim=H3_NN,
               output_dim=10,
               activation=None)
pred = tf.nn.softmax(forward)

# 设置训练参数
train_epochs = 50 # 训练轮数
batch_size = 100 # 单次训练样本数（批次大小）
LEARNING_RATE_STEP = int(mnist.train.num_examples/batch_size) # 一轮训练有多少批次
display_step = 10 # 显示粒度
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
global_step = tf.Variable(0, name='epoch', trainable=False)

# 定义交叉熵损失函数
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=forward, labels=y))

# 学习率
learning_rate = tf.compat.v1.train.exponential_decay( LEARNING_RATE_BASE,
                                                      global_step,
                                                      LEARNING_RATE_STEP,
                                                      LEARNING_RATE_DECAY,
                                                      staircase=True)

# 选择优化器
# 梯度下降优化器
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)
#optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss_function)

# 定义准确率
# 检查预测类别tf.argmax(pred,1)与实际类别tf.argmax(y,1)的匹配情况
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(pred,1))
# 准确率，将布尔值转化为浮点数，并计算平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

from time import time
startTime = time()

# 断点续训
# 创建保存模型文件的目录
ckpt_dir = "/Users/lumin/Documents/dataset/MNIST_data/ckpt_dir/"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
save_step = 10 #存储模型的粒度


# 声明会话
sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer() # 变量初始化
sess.run(init)

#声明完成所有变量后，调用tf.train.Saver
saver = tf.compat.v1.train.Saver()

#如果有检查点文件，读取最新的检查点文件，恢复各种变量值
ckpt = tf.train.get_checkpoint_state(ckpt_dir)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path) #加载所有的参数
    # 从这里开始就可以直接使用模型进行预测,或者接着继续训练了
else:
    print("Traing from scratch.")

# 获取续训参数
start = sess.run(global_step)
print("Trainig starts form {} epoch.".format(start + 1))

# 模型训练
# 开始训练
for ep in range(start, train_epochs):
    for step in range(LEARNING_RATE_STEP):
    
        xs, ys = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict = {x:xs,y:ys})  # 执行批次训练
        
    # total_batch个批次训练完成后,使用验证数据计算误差与准确率;验证集没有分批
    loss,acc = sess.run([loss_function, accuracy], feed_dict={x: mnist.validation.images, y: mnist.validation.labels})
    
    # 打印训练过程中的详细信息
    if(ep+1) % display_step == 0:
        print("Train Epoch:%3d" % (ep+1), "Loss={:.9f}".format(loss), "Accuracy={:.4f}".format(acc))
    
    if (ep+1) % save_step == 0:
        saver.save(sess, os.path.join(ckpt_dir,'mnist_fcn_layer_model_{:06d}.ckpt'.format(ep+1))) # 存储模型
        print('mnist_fcn_layer_model_{:06d}.ckpt saved'.format(ep+1))
        sess.run(global_step.assign(ep+1))
    
saver.save(sess, os.path.join(ckpt_dir, 'mnist_fcn_layer_model.ckpt'))

converter = tf.lite.TFLiteConverter.from_session(sess)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)




print("Model saved!")
#运行总时间
duration = time()-startTime
print("Train Finished takes:","{:.2f}".format(duration))

accu_test = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
print("Test Accuracy:", accu_test)
