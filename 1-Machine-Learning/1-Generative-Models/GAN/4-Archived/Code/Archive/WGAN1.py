import time
import os
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential

from tensorflow.keras import backend
from keras.initializers import RandomNormal
from keras.constraints import Constraint

class ClipConstraint(tf.keras.constraints.Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value

    # clip model weights to hypercube
    def __call__(self, weights):
        return backend.clip(weights, -self.clip_value, self.clip_value)

    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}

def generator_model():
    
    init = RandomNormal(stddev=0.02) # weight initialization

    model = Sequential()

    model.add(Dense(7 * 7 * 128, kernel_initializer=init, input_dim=100))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7, 7, 128)))
 
    # upsample to 14x14
    model.add(Conv2DTranspose(128, (4, 4), strides=(2,2), padding='same', kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
 
    # upsample to 28x28
    model.add(Conv2DTranspose(128, (4, 4), strides=(2,2), padding='same', kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
 
    # output 28x28x1
    model.add(Conv2D(1, (7,7), activation='tanh', padding='same', kernel_initializer=init))

    return model

def discriminator_model():
    
    init = RandomNormal(stddev=0.02) # weight initialization  
    const = ClipConstraint(0.01) # weight constraint

    model = Sequential()
 
    # downsample to 14x14
    model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init, kernel_constraint=const, input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
 
    # downsample to 7x7
    model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init, kernel_constraint=const))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
 
    # scoring, linear activation
    model.add(Flatten())
    model.add(Dense(1))
 
    return model

def discriminator_loss(real_output, fake_output):  
    return -tf.reduce_mean(real_output) + tf.reduce_mean(fake_output)

def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)


generator_optimizer = RMSprop(learning_rate=0.00005)
discriminator_optimizer = RMSprop(learning_rate=0.00005)

################################################################

@tf.function
def train_discriminator(images):

    with tf.GradientTape() as disc_tape:

        noise = tf.random.normal([BATCH_SIZE, noise_dim])
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

@tf.function
def train_generator():

    with tf.GradientTape() as gen_tape:

        noise = tf.random.normal([BATCH_SIZE, noise_dim])
        generated_images = generator(noise, training=True)

        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))


@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    for _ in range(n_critic):

        with tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)
            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)
   
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    with tf.GradientTape() as gen_tape:
        generated_images = generator(noise, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

################################

def generate_and_save_images(model, epoch, test_input):

  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:   
            # for _ in range(n_critic):
            #     train_discriminator(image_batch)
            # train_generator()

            train_step(image_batch)
    
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)

        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # 最后一个 epoch 结束后生成图片
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)

generator = generator_model()
discriminator = discriminator_model()

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16
n_critic=5

(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images[train_labels == 0] # select a given class
train_images = np.expand_dims(train_images, axis=-1).astype('float32')
train_images = (train_images - 127.5) / 127.5

BUFFER_SIZE = train_images.shape[0] # 数据集大小
BATCH_SIZE = 256

# 批量化和打乱数据
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

train(train_dataset, EPOCHS)