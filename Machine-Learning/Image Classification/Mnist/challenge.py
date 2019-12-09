# import tensorflow as tf
import numpy as np
from PIL import Image
import os

# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

x_test = np.genfromtxt('data/challenge/cdigits_digits_vec.txt')
x_test = x_test.reshape(150, 28, 28).astype(np.uint8)
y_test = np.genfromtxt('data/challenge/cdigits_digits_labels.txt')

print(x_test[1])

# my_model = tf.keras.models.load_model('my_model.h5')
# my_model.summary()
# loss, acc = my_model.evaluate(x_test, y_test, verbose=2)

# try:
#     os.mkdir('newtest1')
# except OSError:
#     pass



# for i, image in enumerate(x_test):
#     image = Image.fromarray(image)  # 把数组转换为图像格式  
#     image.save('./%s/%s_%s.bmp' % ('newtest1', 'test', i), 'bmp')  # 储存到文件夹
