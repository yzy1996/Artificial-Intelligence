import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# 获取数据
mnist = input_data.read_data_sets("C:/Users/Administrator/.spyder-py3/MNIST_data/", one_hot=True)

print('训练集信息：')
print(mnist.train.images.shape,mnist.train.labels.shape)
print('测试集信息：')
print(mnist.test.images.shape,mnist.test.labels.shape)
print('验证集信息：')
print(mnist.validation.images.shape,mnist.validation.labels.shape)

# 构建图
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)

y_ = tf.placeholder(tf.float32, [None,10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 进行训练
tf.global_variables_initializer().run()

for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  train_step.run({x: batch_xs, y_: batch_ys})

# 模型评估
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('MNIST手写图片准确率：')
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))