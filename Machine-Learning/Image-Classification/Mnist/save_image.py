import numpy as np
from PIL import Image

x_test = np.genfromtxt('data/challenge/cdigits_digits_vec.txt')
x_test = x_test.reshape(150, 28, 28).astype(np.uint8)
y_test = np.genfromtxt('data/challenge/cdigits_digits_labels.txt')

image = Image.fromarray(x_test[1])  # 把数组转换为图像格式  
image.save('0.bmp', 'bmp')  # 储存到文件夹