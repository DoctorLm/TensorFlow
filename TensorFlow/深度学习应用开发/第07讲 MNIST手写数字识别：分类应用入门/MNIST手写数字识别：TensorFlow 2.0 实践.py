#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
print("TensorFlow版本是:", tf.__version__)

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print("Train image shape:", train_images.shape, "Train label shape:", train_labels.shape)
print("Test image shape:", test_images.shape, "Test label shape:", test_labels.shape)
print("image data", train_images[1])
print("label data", train_labels[1])
type(train_images[1,1,1])

def plot_image(image):
    plt.imshow(image.reshape(28,28), cmap = 'binary')
    plt.show()
plot_image(train_images[1])
plot_image(train_images[20000])

int_array = np.array([i for i in range(64)])
print(int_array)
int_array.reshape(8,8)
int_array.reshape(4,16)
plt.imshow(train_images[20000].reshape(14,56), cmap = 'binary')

total_num = len(train_images)
valid_split = 0.2
train_num = int(total_num*(1-valid_split))
train_x = train_images[:train_num]
train_y = train_labels[:train_num]
valid_x = train_images[train_num:]
valid_y = train_labels[train_num:]
test_x = test_images
test_y = test_labels
valid_x.shape

train_x = train_x.reshape(-1,784)
valid_x = valid_x.reshape(-1,784)
test_x = test_x.reshape(-1,784)

train_x = tf.cast(train_x/255.0, tf.float32)
valid_x = tf.cast(valid_x/255.0, tf.float32)
test_x = tf.cast(test_x/255.0, tf.float32)
train_x[1]

x = [3, 4]
tf.one_hot(x, depth=10)
train_y = tf.one_hot(train_y, depth=10)
valid_y = tf.one_hot(valid_y, depth=10)
test_y = tf.one_hot(test_y, depth=10)
train_y

def model(x, w, b):
    pred = tf.matmul(x, w) + b
    return tf.nn.softmax(pred)
W = tf.Variable(tf.random.normal([784,10], mean=0.0, stddev=1.0, dtype=tf.float32))
B = tf.Variable(tf.zeros([10]), dtype=tf.float32)

def loss(x, y, w, b):
    pred = model(x, w, b)
    loss_ = tf.keras.losses.categorical_crossentropy(y_true=y, y_pred=pred)
    return tf.reduce_mean(loss_)

training_epochs = 20
batch_size = 50
learning_rate = 0.001

def grad(x, y, w, b):
    with tf.GradientTape() as tape:
        loss_ = loss(x, y, w, b)
    return tape.gradient(loss_, [w, b])

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

def accuracy(x, y, w, b):
    pred = model(x, w, b)
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

total_step = int(train_num/batch_size)
loss_list_train = []
loss_list_valid = []
acc_list_train = []
acc_list_valid = []

for epoch in range(training_epochs):
    for step in range(total_step):
        xs = train_x[step*batch_size:(step+1)*batch_size]
        ys = train_y[step*batch_size:(step+1)*batch_size]
        
        grads = grad(xs, ys, W, B)
        optimizer.apply_gradients(zip(grads, [W, B]))
        
    loss_train = loss(train_x, train_y, W, B).numpy()
    loss_valid = loss(valid_x, valid_y, W, B).numpy()
    acc_train = accuracy(train_x, train_y, W, B).numpy()
    acc_valid = accuracy(valid_x, valid_y, W, B).numpy()
    loss_list_train.append(loss_train)
    loss_list_valid.append(loss_valid)
    acc_list_train.append(acc_train)
    acc_list_valid.append(acc_valid)
    print("epoch={:3d},train_loss={:.4f},train_acc={:.4f},val_loss={:.4f},val_acc={:.4f}".format(epoch+1, loss_train, acc_train, loss_valid, acc_valid))

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(loss_list_train, 'blue', label="Train Loss")
plt.plot(loss_list_valid, 'red', label="Valid Loss")
plt.legend(loc=1)

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.plot(acc_list_train, 'blue', label="Train Acc")
plt.plot(acc_list_valid, 'red', label="Valid Acc")
plt.legend(loc=1)

def predict(x, w, b):
    pred = model(x, w, b)
    result = tf.argmax(pred, 1).numpy()
    return result

pred_test = predict(test_x, W, B)
pred_test[0]

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

plot_images_labels_prediction(test_images, test_labels, pred_test, 10, 10)

