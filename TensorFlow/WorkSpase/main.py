import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
print("TensorFlow版本:", tf.__version__)

np.random.seed(5)
x_data = np.linspace(-1, 1, 100)
y_data = 2 * x_data + 1.0 + np.random.randn( * x_data.shape) * 0.4

x = tf.compat.v1.placeholder("float", name = "x")
y = tf.compat.v1.placeholder("float", name = "y")

w = tf.Variable(1.0, name = "w0")
b = tf.Variable(0.0, name = "b0")

def model(x, w, b):
    return tf.multiply(x, w) + b
    
pred = model(x, w, b)

train_epochs = 10
learning_rate = 0.05
display_step = 10

loss_function = tf.reduce_mean(tf.square(y - pred))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

step = 0
loss_list = []
for epoch in range(train_epochs): #10轮
    for xs, ys in zip(x_data, y_data):
        _, loss=sess.run([optimizer, loss_function], feed_dict={x:xs, y:ys})
        loss_list.append(loss)
        step = step + 1
    if step % display_step == 0:
        print("Train Epoch:%2d"%(epoch + 1), "step:%4d"%(step), "loss={:.9f}".format(loss))
    b0temp = b.eval(session = sess)
    w0temp = w.eval(session = sess)
    plt.plot(x_data, w0temp * x_data + b0temp)
plt.show()

plt.plot(loss_list)
plt.show()
plt.plot(loss_list,'g2')
plt.show()

print("w:", sess.run(w)) # W的值应该在2附过
print("b:", sess.run(b)) # b的值应该在1附过

print([x for x in loss_list if x>1])

plt.scatter(x_data, y_data, label = 'Original data')
plt.plot(x_data, x_data * sess.run(w) + sess.run(b), label = 'Fitted line', color = 'r', linewidth = 3)
plt.legend(loc = 2)
plt.show()

x_test = 3.21
predict = sess.run(pred, feed_dict={x: x_test})
print("预测值:%f" % predict)
target = 2 * x_test + 1.0
print("目标值:%f" % target)

predict = sess.run(w) * x_test + sess.run(b)
print("预测值:%f" % predict)
target = 2 * x_test + 1.0
print("目标值:%f" % target)
