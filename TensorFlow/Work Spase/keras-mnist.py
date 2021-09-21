#coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
print("TensorFlow版本:", tf.__version__)
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
              
logdir = './logs'
checkpoint_path = "./checkpoint/mnist.{epoch:02d}-{loss:.2f}.ckpt"

callbacks= [tf.keras.callbacks.TensorBoard(log_dir=logdir,histogram_freq=2),
            tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1, period=1)]

train_history = model.fit(x_train, y_train, epochs=5, callbacks=callbacks, verbose=2)

print(train_history.history)

print(train_history.history.keys ())

evaluate_result = model.evaluate(x_test,  y_test, verbose=2)

print(evaluate_result)

print(model.metrics_names)



