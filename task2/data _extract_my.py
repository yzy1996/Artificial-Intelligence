# 数据集网站：http://yann.lecun.com/exdb/mnist/
# 数据说明：labels 前4个字节是一个magic number，

import struct
import numpy as np
from PIL import Image


# with open('train-labels.idx1-ubyte', 'rb') as f:
#     magic, n = struct.unpack('>ii', f.read(8)) ## 
#     label = np.fromfile(f, dtype=np.uint8)

with open('train-images.idx3-ubyte', 'rb') as f:
    magic_images, num, rows, cols = struct.unpack('>iiii', f.read(16))
    images = np.fromfile(f, dtype=np.uint8)  ## 读后面的字节，现在图片被存到一个数组中
    images = images.reshape(num, 784)  ## 拆分数组，将每张图片单独存到一个数组，（有多少张图片，每张图片大小是28*28=784）
    for image in images:
        image = image.reshape(28, 28)
        image = Image.fromarray(image)
        image.save('./train/train_%s.bmp %image', 'bmp')
# print(len(image))