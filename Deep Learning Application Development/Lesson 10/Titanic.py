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

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(units=64,
                                input_dim=7,
                                use_bias=True,
                                kernel_initializer='uniform',
                                bias_initializer='zeros',
                                activation='relu'))
                                
model.add(tf.keras.layers.Dense(units=32, activation='sigmoid'))

model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(0.003),
              loss='binary_crossentropy',
              metrics=['accuracy'])

print(dir(model.fit))
train_history=model.fit(x=x_train,
                        y=y_train,
                        validation_split=0.2,
                        epochs=100,
                        batch_size=40,
                        verbose=2)
