'''using cnn of LeNet-5 to train on Mnist
'''

import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# standard mnist dataset
# 元组 x_train, x_test: uint8 数组表示的灰度图像，尺寸为 (num_samples, 28, 28)。
# 元组 y_train, y_test: uint8 数组表示的数字标签（范围在 0-9 之间的整数），尺寸为 (num_samples,)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((-1,28,28,1)) / 255.0
x_test = x_test.reshape((-1,28,28,1)) / 255.0

model = tf.keras.models.Sequential()

# C1 Convolutional Layer
model.add(layers.Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='same', input_shape=(28, 28, 1)))

# S2 Pooling Layer
model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))

# C3 Convolutional Layer
model.add(layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))

# S4 Pooling Layer
model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

# C5 Fully Connected Convolutional Layer
model.add(layers.Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))

#Flatten the CNN output so that we can connect it with fully connected layers
model.add(layers.Flatten())

# FC6 Fully Connected Layer
model.add(layers.Dense(84, activation='tanh'))

model.add(layers.Dropout(0.2))

#Output Layer with softmax activation
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

hist = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.1, verbose=2)
model.save('my_model.h5')
test_score = model.evaluate(x_test,  y_test, verbose=2)

f, ax = plt.plot()
ax.plot([None] + hist.history['accuracy'], 'o-')
ax.legend(['Train acc'], loc = 0)
ax.set_title('Training accuracy per Epoch')
ax.set_xlabel('Epoch')
ax.set_ylabel('accuracy')
plt.show()