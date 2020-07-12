import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np 
import random

'''
python 3.7
tensorflow 2.0.0b0
'''


class CNN(object):
    def __init__(self):
        model = models.Sequential()
        # 第1层卷积，卷积核大小为3*3，32个，28*28为待训练图片的大小
        model.add(layers.Conv2D(
            32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        # 第2层卷积，卷积核大小为3*3，64个
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        # 第3层卷积，卷积核大小为3*3，64个
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))

        model.summary()

        self.model = model


class DataSource(object):
    def __init__(self):
        # mnist数据集存储的位置，如何不存在将自动下载
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        for i in range(len(x_train)):
            x_train[i] = np.rot90(x_train[i], random.randint(0,4))

        x_test = np.genfromtxt('../data/challenge/cdigits_digits_vec.txt').reshape(150, 28, 28)
        y_test = np.genfromtxt('../data/challenge/cdigits_digits_labels.txt')

        x_train = x_train.reshape((60000, 28, 28, 1))
        x_test = x_test.reshape((150, 28, 28, 1))

        x_train, x_test = x_train / 255.0, x_test / 255.0

        self.train_images, self.train_labels = x_train, y_train
        self.test_images, self.test_labels = x_test, y_test

class Train:
    def __init__(self):
        self.cnn = CNN()
        self.data = DataSource()

    def train(self):
        check_path = './ckpt/cp-{epoch:04d}.ckpt'
        # period 每隔5epoch保存一次
        save_model_cb = tf.keras.callbacks.ModelCheckpoint(
            check_path, save_weights_only=True, verbose=1, period=5)

        self.cnn.model.compile(optimizer='adam',
                               loss='sparse_categorical_crossentropy',
                               metrics=['accuracy'])
        self.cnn.model.fit(self.data.train_images, self.data.train_labels,
                           epochs=5, callbacks=[save_model_cb])

        test_loss, test_acc = self.cnn.model.evaluate(
            self.data.test_images, self.data.test_labels)
        print("准确率: %.4f，共测试了%d张图片 " % (test_acc, len(self.data.test_labels)))


if __name__ == "__main__":
    app = Train()
    app.train()