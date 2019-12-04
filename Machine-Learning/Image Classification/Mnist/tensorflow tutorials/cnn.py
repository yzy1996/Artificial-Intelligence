import tensorflow as tf
from nn import X_train, X_validation, y_train, y_validation
import numpy as np
import random
tf.compat.v1.disable_eager_execution()
sess = tf.compat.v1.InteractiveSession()

for i in range(len(X_train)):
    X_train[i] = np.rot90(X_train[i].reshape(28, 28), random.randint(0,4)).reshape(784)

x_test = np.genfromtxt('../data/challenge/cdigits_digits_vec.txt')
yy = np.genfromtxt('../data/challenge/cdigits_digits_labels.txt').astype(np.int32)

test_data = []
for label in yy:
    zero_vector = np.zeros((1, 10))
    zero_vector[0, label] = 1
    test_data.append(zero_vector)

y_test = [i[0] for i in test_data]

def weight_variable(shape):
    initial = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(input=x, filters=W, strides=[1, 1, 1, 1], padding='SAME')  



def max_pool_2x2(x):
    return tf.nn.max_pool2d(input=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x = tf.compat.v1.placeholder(tf.float32, [None, 784], name='x-input')
y_ = tf.compat.v1.placeholder(tf.float32, [None,10], name='y-input')
x_image = tf.reshape(x, [-1, 28, 28, 1])  


W_conv1 = weight_variable([5, 5, 1, 6])  
b_conv1 = bias_variable([6])
h_conv1 = tf.nn.leaky_relu(conv2d(x_image, W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1) 


W_conv2 = weight_variable([5, 5, 6, 12])  
b_conv2 = bias_variable([12])
h_conv2 = tf.nn.leaky_relu(conv2d(h_pool1, W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)  


W_fc1 = weight_variable([7*7*12, 200])
b_fc1 = bias_variable([200])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*12])
h_fc1 = tf.nn.leaky_relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)


keep_prob = tf.compat.v1.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, 1 - (keep_prob))

W_fc2 = weight_variable([200, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2)+b_fc2)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_conv, labels=tf.argmax(input=y_, axis=1))
cross_entropy = tf.reduce_mean(input_tensor=cross_entropy)
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),
#                                               reduction_indices=[1]))


correct_prediction = tf.equal(tf.argmax(input=y_conv, axis=1), tf.argmax(input=y_, axis=1))
accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))
learning_rate_base = 0.005
learning_rate_decay = 0.999
batch_size = 200
global_step = tf.Variable(0, trainable=False)

# learning_rate = tf.train.exponential_decay(learning_rate_base,
#                                            global_step,
#                                            len(X_train)/batch_size,
#                                            learning_rate_decay)
learning_rate = 0.001
train_step = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

tf.compat.v1.global_variables_initializer().run()

for i in range(10000):
    if i % 200 == 0:
        validation_dict = {x: X_validation, y_: y_validation, keep_prob: 1.0}
        val_accuracy = accuracy.eval(feed_dict=validation_dict)
        print("step %d, validation accuracy is %g" % (i, val_accuracy))
        # print(cross_entropy.eval())

    start = (i*batch_size) % len(X_train)
    end = min(start+batch_size, len(X_train))
    train_step.run(feed_dict={x: X_train[start:end], y_: y_train[start:end], keep_prob: 0.5})

# accuracy.save('my_model.h5')
test_accuracy = accuracy.eval(feed_dict={x: x_test, y_: y_test, keep_prob: 1.0})
print("test accuracy is %g" % test_accuracy)