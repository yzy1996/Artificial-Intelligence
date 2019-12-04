# import tensorflow as tf
import numpy as np
# import tensorflow as tf

x_test = np.genfromtxt('data/challenge/cdigits_digits_vec.txt')
x_test = x_test.reshape(150, 28, 28)
y_test = np.genfromtxt('data/challenge/cdigits_digits_labels.txt')

yy = []
for label in y_test:
    zero_vector = np.zeros((1, 10))
    zero_vector[0, label] = 1
    yy.append(zero_vector)

y_test = [i[0] for i in yy]
print(type(y_test[0]))
# print(np.shape(y_test))

# my_model = tf.keras.models.load_model('my_model.h5')
# my_model.evaluate(x_test, y_test, verbose=2)
