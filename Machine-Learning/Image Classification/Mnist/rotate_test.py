# import tensorflow as tf
import numpy as np
from PIL import Image
import os
import random

# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

x_test = np.genfromtxt('data/challenge/cdigits_digits_vec.txt')
x_test = x_test.reshape(150, 28, 28).astype(np.uint8)
y_test = np.genfromtxt('data/challenge/cdigits_digits_labels.txt')

image = Image.fromarray(x_test[1])  # 把数组转换为图像格式  
image.save('0.bmp', 'bmp')  # 储存到文件夹

# 左右翻转
image = x_test[2][:,::-1]
image = np.rot90(image, axes=(0,1))
# image = np.rot90(x_test[1], random.randint(0,4))

image = Image.fromarray(image)  # 把数组转换为图像格式  
image.save('1.bmp', 'bmp')  # 储存到文件夹
