'''
CelebA_WGAN-GP_Tensorflow

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

FILTER_NUM = 32
NOISE_DIM = 100

NUM_TO_GENERATE = 5  # square


# 创建输出文件夹
OUTPUT_PATH = Path('./Output_CelebA_WGAN-GP_Tnesorflow')

IMG_PATH = Path('D:/Data/Face/celeba/Male/positive/')

# 定义网络模型
def make_generator_model():

    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    model = tf.keras.Sequential()

    model.add(tf.keras.Input(shape=(NOISE_DIM, )))

    model.add(layers.Dense(4 * 4 * FILTER_NUM * 8, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((4, 4, FILTER_NUM * 8)))

    model.add(layers.Conv2DTranspose(filters=FILTER_NUM * 8,
                                     kernel_size=5,
                                     strides=1,
                                     padding='same',
                                     use_bias=False,
                                     kernel_initializer=initializer))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(filters=FILTER_NUM * 4,
                                     kernel_size=5,
                                     strides=2,
                                     padding='same',
                                     use_bias=False,
                                     kernel_initializer=initializer))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(filters=FILTER_NUM * 2,
                                     kernel_size=5,
                                     strides=2,
                                     padding='same',
                                     use_bias=False,
                                     kernel_initializer=initializer))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(filters=FILTER_NUM,
                                     kernel_size=5,
                                     strides=2,
                                     padding='same',
                                     use_bias=False,
                                     kernel_initializer=initializer))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(filters=3,
                                     kernel_size=5,
                                     strides=2,
                                     padding='same',
                                     use_bias=False,
                                     activation='tanh',
                                     kernel_initializer=initializer))

    return model


def make_discriminator_model():

    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    model = tf.keras.Sequential()

    model.add(tf.keras.Input(shape=(64, 64, 3)))

    model.add(layers.Conv2D(filters=FILTER_NUM,
                            kernel_size=5,
                            strides=2,
                            padding='same',
                            kernel_initializer=initializer))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(filters=FILTER_NUM * 2,
                            kernel_size=5,
                            strides=2,
                            padding='same',
                            kernel_initializer=initializer))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(filters=FILTER_NUM * 4,
                            kernel_size=5,
                            strides=2,
                            padding='same',
                            kernel_initializer=initializer))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(filters=FILTER_NUM * 8,
                            kernel_size=5,
                            strides=2,
                            padding='same',
                            kernel_initializer=initializer))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

# Gradient Penalty (GP)
def gradient_penalty(f, real_data, fake_data):

    shape = [real_data.shape[0]] + [1] * (real_data.shape.ndims - 1)
    alpha = tf.random.uniform(shape=shape)
    
    interpolate = real_data + alpha * (fake_data - real_data)
    
    with tf.GradientTape() as tape:
        tape.watch(interpolate)
        interpolate_output = f(interpolate)
        
    gradient = tape.gradient(interpolate_output, interpolate)
    norm = tf.norm(tf.reshape(gradient, [tf.shape(gradient)[0], -1]), axis=1)

    return tf.reduce_mean((norm - 1.) ** 2)


def discriminator_loss_fn(real_output, fake_output):
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)


def generator_loss_fn(fake_output):
    return -tf.reduce_mean(fake_output)


generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0, beta_2=0.9)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0, beta_2=0.9)

generator_loss_measure = tf.keras.metrics.Mean(name='generator_loss')
discriminator_loss_measure = tf.keras.metrics.Mean(name='discriminator_loss')


@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    for _ in range(5):

        # ---------------------
        #  Train Discriminator
        # ---------------------
        with tf.GradientTape() as disc_tape:

            generated_images = generator(noise, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)
   
            discriminator_loss = discriminator_loss_fn(real_output, fake_output)

            gp = gradient_penalty(discriminator, images, generated_images)
            discriminator_loss += gp * 10

            discriminator_loss_measure.update_state(discriminator_loss)

        discriminator_gradient = disc_tape.gradient(discriminator_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(discriminator_gradient, discriminator.trainable_variables))

    # ---------------------
    #  Train Generator
    # ---------------------
    with tf.GradientTape() as gen_tape:

        generated_images = generator(noise, training=True)

        fake_output = discriminator(generated_images, training=True)

        generator_loss = generator_loss_fn(fake_output)
        
        generator_loss_measure.update_state(generator_loss)

    generator_gradient = gen_tape.gradient(generator_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradient, generator.trainable_variables))


def generate_and_save_images(predictions, epoch):

    IMAGE_PATH = OUTPUT_PATH / 'image'

    if not IMAGE_PATH.exists():
        IMAGE_PATH.mkdir(parents=True)

    fig = plt.figure(figsize=(NUM_TO_GENERATE, NUM_TO_GENERATE))

    for i in range(predictions.shape[0]):
        img = (predictions[i] + 1) / 2
        plt.subplot(NUM_TO_GENERATE, NUM_TO_GENERATE, i + 1)
        plt.imshow(img)
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


def load_image(img_path):
    img = tf.io.read_file(img_path) 
    img = tf.image.decode_jpeg(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    img = tf.image.central_crop(img, 0.7)
    # img = tf.image.resize_with_crop_or_pad(img, 128, 128)
    img = tf.image.resize(img, [64, 64])
    img = img * 2 - 1
    return img


train_dataset = tf.data.Dataset.list_files('D:/Data/Face/celeba/Male/positive/*.jpg') \
    .map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    .batch(BATCH_SIZE, drop_remainder=True)

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

generator.save(OUTPUT_PATH / 'CelebA_WGAN-GP.h5')
