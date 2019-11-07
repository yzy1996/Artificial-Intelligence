# 数据集网站：http://yann.lecun.com/exdb/mnist/
# 数据说明：
# labels 前4个字节是 magic number，接着的4个字节是 number of items，之后每一个字节存一个数
# images 前4个字节是 magic number，接着的4个字节是 number of items，再4个字节是 number of rows，再4个字节是 number of columns，之后每一个字节存一个数

import os
import struct
import numpy as np
from PIL import Image

# struct.unpack()括号第一个参数表示按什么 fmt 来解读，第二个表示读几字节，因为整数是32位4字节，所以ii（两整数）需要8字节来读取，这两个是搭配关系，必须一致，>代表了从大到小排列

def extract_mnist_label(filename, type):
    with open(filename, 'rb') as f:
        magic_labels, num = struct.unpack('>ii', f.read(8))   
        labels = np.fromfile(f, dtype=np.uint8)  # python的read是接着读，相当于正好现在开始读核心数据

    np.savetxt('./%s_labels.txt' %type, labels, fmt='%i', delimiter=',')

def extract_mnist_image(filename, type):

    try:
        os.mkdir(type)
    except OSError:
        pass

    with open(filename, 'rb') as f:
        magic_images, num, rows, cols = struct.unpack('>iiii', f.read(16))
        images = np.fromfile(f, dtype=np.uint8)  ## 读后面的字节，现在图片被存到一个数组中
        images = images.reshape(num, 28, 28)  ## 拆分数组，将每张图片单独存到一个数组，（有多少张图片，每张图片大小是28*28=784）

        for i, image in enumerate(images):       
            image = Image.fromarray(image)  # 把数组转换为图像格式
            image.save('./%s/%s_%s.bmp' %(type, type, i), 'bmp')  # 储存到文件夹

## images 存的是一个三维数组，60000*28*28，images[i]就是一个28*28的单张图片像素值 

if __name__ == '__main__':

    train_images = 'train-images.idx3-ubyte'  
    extract_mnist_image(train_images, 'train')

    train_labels = 'train-labels.idx1-ubyte'
    extract_mnist_label(train_labels, 'train')

    test_images = 't10k-images.idx3-ubyte'
    extract_mnist_image(test_images, 'test')

    test_labels = 't10k-labels.idx1-ubyte'
    extract_mnist_label(test_labels, 'test')