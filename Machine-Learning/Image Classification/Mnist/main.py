'''Train and Test Mnist dataset
Build the code step by step

python-3.7; tensorfolow-2.0.0
'''

import struct
import numpy as np
import tensorflow as tf

# prepare work
print("Tensorflow version " + tf.__version__)

# load and prepare the mnist dataset
def extract_mnist_image(filename, type):

    with open(filename, 'rb') as f:
        magic_images, num, rows, cols = struct.unpack('>iiii', f.read(16))
        images = np.fromfile(f, dtype=np.uint8)  # 读后面的字节，现在图片被存到一个数组中
        # 拆分数组，将每张图片单独存到一个数组，（有多少张图片，每张图片大小是28*28=784）
        images = images.reshape(num, 28, 28)

    return images


def extract_mnist_label(filename, type):

    with open(filename, 'rb') as f:
        magic_labels, num = struct.unpack('>ii', f.read(8))
        # python的read是接着读，相当于正好现在开始读核心数据
        labels = np.fromfile(f, dtype=np.uint8)

    return labels


if __name__ == '__main__':

    train_images = 'data/train-images.idx3-ubyte'
    x_train = extract_mnist_image(train_images, 'train')  # (60000, 28, 28)

    train_labels = 'data/train-labels.idx1-ubyte'
    y_train = extract_mnist_label(train_labels, 'train')  # (60000,)

    test_images = 'data/t10k-images.idx3-ubyte'
    x_test = extract_mnist_image(test_images, 'test')  # (10000, 28, 28)

    test_labels = 'data/t10k-labels.idx1-ubyte'
    y_test = extract_mnist_label(test_labels, 'test')  # (10000,)
