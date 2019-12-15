import tensorflow as tf
import numpy as np 


x_test = np.genfromtxt('../data/challenge/cdigits_digits_vec.txt').reshape(150, 28, 28)
y_test = np.genfromtxt('../data/challenge/cdigits_digits_labels.txt')

model = tf.keras.models.load_model('my_model.h5')

num = 149

image = np.expand_dims(x_test[num], axis=0)
predict = model.predict(image)
predict = np.argmax(predict)

print('predict: ', predict)
print('truth: ', y_test[num])