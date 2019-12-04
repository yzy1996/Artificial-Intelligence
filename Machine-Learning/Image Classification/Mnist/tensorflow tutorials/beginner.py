import tensorflow as tf
import numpy as np 
import random


mnist = tf.keras.datasets.mnist

# 元组 x_train, x_test: uint8 数组表示的灰度图像，尺寸为 (num_samples, 28, 28)。
# 元组 y_train, y_test: uint8 数组表示的数字标签（范围在 0-9 之间的整数），尺寸为 (num_samples,)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train[1])
for i in range(len(x_train)):
    x_train[i] = np.rot90(x_train[i], random.randint(0,4))

x_test = np.genfromtxt('../data/challenge/cdigits_digits_vec.txt').reshape(150, 28, 28)
y_test = np.genfromtxt('../data/challenge/cdigits_digits_labels.txt')

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

model.fit(x_train, y_train, epochs=5)
# model.save('my_model.h5')
model.evaluate(x_test,  y_test, verbose=2)

# accuracy: 0.9793