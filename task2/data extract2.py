import struct
import numpy as np






def extract_mnist_image(filename, type):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack('>ii', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)



def extract_mnist_label(filename, type):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>iiii', f.read(16))
        labels = np.fromfile(f, dtype=np.uint8)

if __name__ == '__main__':
    path = 'D:/Project/an_python/Minst/'

    train_images = 'train-images.idx3-ubyte'  
    extract_mnist_image(train_images, 'train')

    train_labels = 'train-labels.idx1-ubyte'
    extract_mnist_label(train_labels, 'train')

    test_images = 't10k-images.idx3-ubyte'
    extract_mnist_image(test_images, 'test')

    test_labels = 't10k-labels.idx1-ubyte'
    extract_mnist_label(test_labels, 'test')