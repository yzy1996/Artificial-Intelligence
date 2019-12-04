
import numpy as np
import struct
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
import random
tf.compat.v1.disable_eager_execution()

class LoadData(object):
    def __init__(self, file1, file2):
        self.file1 = file1
        self.file2 = file2


    def loadImageSet(self):
        binfile = open(self.file1, 'rb') 
        buffers = binfile.read()  
        head = struct.unpack_from('>IIII', buffers, 0)  
        offset = struct.calcsize('>IIII') 
        imgNum = head[1]  
        width = head[2]  
        height = head[3]  

        bits = imgNum*width*height 
        bitsString = '>' + str(bits) + 'B' 
        imgs = struct.unpack_from(bitsString, buffers, offset)  

        binfile.close()
        imgs = np.reshape(imgs, [imgNum, width*height])
        return imgs, head


    def loadLabelSet(self):
        binfile = open(self.file2, 'rb')  
        buffers = binfile.read()  
        head = struct.unpack_from('>II', buffers, 0)  
        offset = struct.calcsize('>II')  

        labelNum = head[1]  
        numString = '>' + str(labelNum) + 'B'
        labels = struct.unpack_from(numString, buffers, offset)  

        binfile.close()
        labels = np.reshape(labels, [labelNum])  
        return labels, head


    def expand_lables(self):
        labels, head = self.loadLabelSet()
        expand_lables = []
        for label in labels:
            zero_vector = np.zeros((1, 10))
            zero_vector[0, label] = 1
            expand_lables.append(zero_vector)
        return expand_lables

    
    def loadData(self):
        imags, head = self.loadImageSet()
        expand_lables = self.expand_lables()
        data = []
        for i in range(imags.shape[0]):
            imags[i] = imags[i].reshape((1, 784))
            data.append([imags[i], expand_lables[i]])
        return data


file1 = r'../data/train-images.idx3-ubyte'
file2 = r'../data/train-labels.idx1-ubyte'
trainingData = LoadData(file1, file2)
training_data = trainingData.loadData()

X_train = [i[0] for i in training_data]
y_train = [i[1][0] for i in training_data]

for i in range(len(X_train)):
    X_train[i] = np.rot90(X_train[i].reshape(28, 28), random.randint(1,3)).reshape(784)

x_test = np.genfromtxt('../data/challenge/cdigits_digits_vec.txt')
yy = np.genfromtxt('../data/challenge/cdigits_digits_labels.txt').astype(np.int32)

test_data = []
for label in yy:
    zero_vector = np.zeros((1, 10))
    zero_vector[0, label] = 1
    test_data.append(zero_vector)

y_test = [i[0] for i in test_data]


X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.1, random_state=7)

INUPUT_NODE = 784
OUTPUT_NODE = 10

LAYER1_NODE = 500
BATCH_SIZE = 200
LERANING_RATE_BASE = 0.005  
LERANING_RATE_DACAY = 0.99  
REGULARZATION_RATE = 0.01  
TRAINING_STEPS = 100000
MOVING_AVERAGE_DECAY = 0.99  



def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    if not avg_class:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1)+biases1)

        return tf.matmul(layer1, weights2)+biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1))+
                            avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2))+avg_class.average(biases2)


def train(X_train, X_validation, y_train, y_validation, X_test, y_test):
    x = tf.compat.v1.placeholder(tf.float32, [None, INUPUT_NODE], name="x-input")
    y_ = tf.compat.v1.placeholder(tf.float32, [None, OUTPUT_NODE], name="y-input")


    weights1 = tf.Variable(
        tf.random.truncated_normal([INUPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))


    weights2 = tf.Variable(
        tf.random.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
    y = inference(x, None, weights1, biases1, weights2, biases2)

    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.compat.v1.trainable_variables())
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(input=y_, axis=1))
    cross_entropy_mean = tf.reduce_mean(input_tensor=cross_entropy)


    regularizer = tf.keras.regularizers.l2(0.5 * (REGULARZATION_RATE))
    regularization = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularization

    learning_rate = tf.compat.v1.train.exponential_decay(LERANING_RATE_BASE,
                                               global_step,
                                               len(X_train)/BATCH_SIZE,
                                               LERANING_RATE_DACAY)
    train_step = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    correct_prediction = tf.equal(tf.argmax(input=average_y, axis=1), tf.argmax(input=y_, axis=1))
    accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))

    with tf.compat.v1.Session() as sess:
        init_op = tf.compat.v1.global_variables_initializer()
        sess.run(init_op)
        validation_feed = {x: X_validation, y_: y_validation}
        train_feed = {x: X_train, y_: y_train}
        test_feed = {x: X_test, y_: y_test}

        for i in range(TRAINING_STEPS):
            if i % 500 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validation_feed)
                print("after %d training step(s), validation accuracy "
                      "using average model is %g" % (i, validate_acc))
            start = (i * BATCH_SIZE) % len(X_train)
            end = min(start + BATCH_SIZE, len(X_train))
            sess.run(train_op,
                     feed_dict={x: X_train[start:end], y_: y_train[start:end]})
            # print('loss:', sess.run(loss))

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("after %d training step(s), test accuracy using"
              "average model is %g" % (TRAINING_STEPS, test_acc))

train(X_train, X_validation, y_train, y_validation, x_test, y_test)