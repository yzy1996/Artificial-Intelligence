import numpy as np 

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
y_test1 = np.genfromtxt('../data/challenge/cdigits_digits_labels.txt').astype(np.uint8)