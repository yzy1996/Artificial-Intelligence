from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

def compute_accuracy(v_x, v_y):
    global prediction
    y_pre = sess.run(prediction, feed_dict={x:v_x, keep_prob:1})
    correct_prediction = tf.equal(tf.argmax(input=y_pre,axis=1), tf.argmax(input=v_y,axis=1))
    accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy,feed_dict={x: v_x, y: v_y, keep_prob:1})
    return result

def weight_variable(shape):
    initial = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
    
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # strides=[1,x_movement,y_movement,1]
    return tf.nn.conv2d(input=x, filters=W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool2d(input=x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# load mnist data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.compat.v1.placeholder(tf.float32, [None,784])
y = tf.compat.v1.placeholder(tf.float32, [None,10])
keep_prob = tf.compat.v1.placeholder(tf.float32)
# reshape(data you want to reshape, [-1, reshape_height, reshape_weight, imagine layers]) image layers=1 when the imagine is in white and black, =3 when the imagine is RGB 
x_image = tf.reshape(x, [-1,28,28,1])

# ********************** conv1 *********************************
# transfer a 5*5*1 imagine into 32 sequence
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
# input a imagine and make a 5*5*1 to 32 with stride=1*1, and activate with relu
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28*28*32
h_pool1 = max_pool_2x2(h_conv1) # output size 14*14*32

# ********************** conv2 *********************************
# transfer a 5*5*32 imagine into 64 sequence
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
# input a imagine and make a 5*5*32 to 64 with stride=1*1, and activate with relu
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14*14*64
h_pool2 = max_pool_2x2(h_conv2) # output size 7*7*64

# ********************* func1 layer *********************************
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
# reshape the image from 7,7,64 into a flat (7*7*64)
h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1,1 - (keep_prob))

# ********************* func2 layer *********************************
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# calculate the loss
cross_entropy = tf.reduce_mean(input_tensor=-tf.reduce_sum(input_tensor=y*tf.math.log(prediction), axis=[1]))
# use Gradientdescentoptimizer
train_step = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# init session
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for i in range(1000):
    batch_x, batch_y = x_train.next_batch(100)
    sess.run(train_step,feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))
