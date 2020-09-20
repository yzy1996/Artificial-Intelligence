import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

img_path = 'D:/Data/Face/celeba/Male/positive/000003.jpg'
img_path2 = 'D:/Data/Face/celeba/Male/positive/000007.jpg'

img = tf.io.read_file(img_path)
img = tf.image.decode_jpeg(img)
img = tf.image.resize(img, [64, 64])
img /= 255.0


print(img)