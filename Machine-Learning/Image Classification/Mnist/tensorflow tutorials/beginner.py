import tensorflow as tf
import numpy as np 
import random

# # standard mnist dataset
# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# # 元组 x_train, x_test: uint8 数组表示的灰度图像，尺寸为 (num_samples, 28, 28)。
# # 元组 y_train, y_test: uint8 数组表示的数字标签（范围在 0-9 之间的整数），尺寸为 (num_samples,)


# sequence of the image and label
train_seq = np.genfromtxt('../data/digits4000_txt/digits4000_trainset.txt').astype(np.uint16) # (2000,2)
test_seq = np.genfromtxt('../data/digits4000_txt/digits4000_testset.txt').astype(np.uint16) # (2000,2)

# image and label
digits_vec = np.genfromtxt('../data/digits4000_txt/digits4000_digits_vec.txt') # (4000,28,28)
digits_vec = digits_vec.reshape(len(digits_vec), 28, 28).astype(np.uint8)
digits_labels = np.genfromtxt('../data/digits4000_txt/digits4000_digits_labels.txt').astype(np.uint8) # (4000,)

x_train = digits_vec[train_seq[:,0] - 1]
y_train = digits_labels[train_seq[:,1] - 1]

x_test = digits_vec[test_seq[:,0] - 1]
y_test = digits_labels[test_seq[:,1] - 1]

# challenge test image and label
x_test1 = np.genfromtxt('../data/challenge/cdigits_digits_vec.txt')
x_test1 = x_test1.reshape(len(x_test1), 28, 28).astype(np.uint8)
y_test1 = np.genfromtxt('../data/challenge/cdigits_digits_labels.txt')


x_train, x_test, x_test1 = x_train / 255.0, x_test / 255.0, x_test1 / 255.0

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(10, activation='softmax'))


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)
# model.save('my_model.h5')
model.evaluate(x_test,  y_test, verbose=2)
model.evaluate(x_test1,  y_test1, verbose=2)

# accuracy: 0.9793