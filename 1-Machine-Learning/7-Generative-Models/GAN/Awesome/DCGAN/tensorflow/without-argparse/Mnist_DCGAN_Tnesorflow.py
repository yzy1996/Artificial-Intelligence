'''
Mnist_DCGAN_Tnesorflow

by Zhiyuan Yang
'''

import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from pathlib import Path
import time


# 定义常量 Constant
EPOCHS = 50
BATCH_SIZE = 128
BUFFER_SIZE = 60000

FILTER_NUM = 128
NOISE_DIM = 100

NUM_TO_GENERATE = 5  # square


# 创建输出文件夹
OUTPUT_PATH = Path('./Output_Mnist_DCGAN_Tnesorflow')


# 定义网络模型
def make_generator_model():

    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(NOISE_DIM, )))

    model.add(layers.Dense(7 * 7 * FILTER_NUM * 2, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((7, 7, FILTER_NUM * 2)))

    model.add(layers.Conv2DTranspose(filters=FILTER_NUM * 2,
                                     kernel_size=4,
                                     strides=1,
                                     padding='same',
                                     use_bias=False,
                                     kernel_initializer=initializer))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(filters=FILTER_NUM,
                                     kernel_size=4,
                                     strides=2,
                                     padding='same',
                                     use_bias=False,
                                     kernel_initializer=initializer))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(filters=1,
                                     kernel_size=4,
                                     strides=(2, 2),
                                     padding='same',
                                     use_bias=False,
                                     activation='tanh',
                                     kernel_initializer=initializer))

    return model


def make_discriminator_model():

    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(28, 28, 1)))

    model.add(layers.Conv2D(filters=FILTER_NUM,
                            kernel_size=4,
                            strides=2,
                            padding='same',
                            kernel_initializer=initializer))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(filters=FILTER_NUM * 2,
                            kernel_size=4,
                            strides=2,
                            padding='same',
                            kernel_initializer=initializer))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())
    model.add(layers.Dense(units=1))

    return model


# 定义损失函数
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss_fn(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return (real_loss + fake_loss) / 2 


def generator_loss_fn(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

generator_loss_measure = tf.keras.metrics.Mean(name='generator_loss')
discriminator_loss_measure = tf.keras.metrics.Mean(name='discriminator_loss')


@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        generator_loss = generator_loss_fn(fake_output)
        discriminator_loss = discriminator_loss_fn(real_output, fake_output)

        generator_loss_measure.update_state(generator_loss)
        discriminator_loss_measure.update_state(discriminator_loss)

    generator_gradient = gen_tape.gradient(generator_loss, generator.trainable_variables)
    discriminator_gradient = disc_tape.gradient(discriminator_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradient, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradient, discriminator.trainable_variables))


def generate_and_save_images(predictions, epoch):

    IMAGE_PATH = OUTPUT_PATH / 'image'

    if not IMAGE_PATH.exists():
        IMAGE_PATH.mkdir(parents=True)

    fig = plt.figure(figsize=(NUM_TO_GENERATE, NUM_TO_GENERATE))

    for i in range(predictions.shape[0]):
        plt.subplot(NUM_TO_GENERATE, NUM_TO_GENERATE, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig(IMAGE_PATH / f'image_at_epoch_{epoch:02d}.png')
    plt.close()


def train(dataset, epochs):

    for epoch in range(epochs):

        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        predictions = generator(seed, training=False)
        generate_and_save_images(predictions, epoch + 1)

        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        tf.print(f'[{epoch + 1}/{EPOCHS}], G_loss = {generator_loss_measure.result():.3f}, D_loss = {discriminator_loss_measure.result():.3f}, time = {time.time() - start:.3f}')

        
        generator_loss_measure.reset_states()
        discriminator_loss_measure.reset_states()


(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

generator = make_generator_model()
discriminator = make_discriminator_model()

checkpoint_dir = OUTPUT_PATH / 'checkpoints'
checkpoint_prefix = checkpoint_dir / 'ckpt'
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

seed = tf.random.normal([NUM_TO_GENERATE ** 2, NOISE_DIM])

train(train_dataset, EPOCHS)

generator.save(OUTPUT_PATH / '/model_DCGAN.h5')
