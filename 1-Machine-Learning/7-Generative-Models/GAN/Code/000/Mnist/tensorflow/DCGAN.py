import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras import layers
import time

EPOCHS = 50
BATCH_SIZE = 128
filters_num = 128
num_square_examples_to_generate = 5
noise_dim = 100


sample_save_folder = './output_DCGAN'


def make_generator_model():

    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(noise_dim, )))

    model.add(layers.Dense(7 * 7 * filters_num * 2, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((7, 7, filters_num * 2)))

    model.add(layers.Conv2DTranspose(filters=filters_num * 2,
                                     kernel_size=4,
                                     strides=1,
                                     padding='same',
                                     use_bias=False,
                                     kernel_initializer=initializer))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(filters=filters_num,
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

    model.add(layers.Conv2D(filters=filters_num,
                            kernel_size=4,
                            strides=2,
                            padding='same',
                            kernel_initializer=initializer))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(filters=filters_num * 2,
                            kernel_size=4,
                            strides=2,
                            padding='same',
                            kernel_initializer=initializer))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())
    model.add(layers.Dense(units=1))

    return model


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss_fn(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss_fn(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

generator_loss_measure = tf.keras.metrics.Mean(name='generator_loss')
discriminator_loss_measure = tf.keras.metrics.Mean(name='discriminator_loss')


@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        generator_loss = generator_loss_fn(fake_output)
        discriminator_loss = discriminator_loss_fn(real_output, fake_output)

        generator_loss_measure.update_state(generator_loss)
        discriminator_loss_measure.update_state(discriminator_loss)

    generator_gradient = gen_tape.gradient(
        generator_loss, generator.trainable_variables)
    discriminator_gradient = disc_tape.gradient(
        discriminator_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(
        zip(generator_gradient, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(
        zip(discriminator_gradient, discriminator.trainable_variables))


def generate_and_save_images(model, epoch, test_input):

    if not os.path.exists(sample_save_folder):
        os.makedirs(sample_save_folder)

    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(num_square_examples_to_generate,
                              num_square_examples_to_generate))

    for i in range(predictions.shape[0]):
        plt.subplot(num_square_examples_to_generate,
                    num_square_examples_to_generate, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig(sample_save_folder + f'/image_at_epoch_{epoch:04d}.png')
    plt.close()


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        generate_and_save_images(generator, epoch + 1, seed)

        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        logs = 'Epoch={}, Time={}, Generator_Loss:{}, Discriminator_Loss:{}'
        tf.print(tf.strings.format(logs, (epoch + 1, time.time() - start, generator_loss_measure.result(), discriminator_loss_measure.result())))

        generator_loss_measure.reset_states()
        discriminator_loss_measure.reset_states()


(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).batch(BATCH_SIZE)


generator = make_generator_model()
discriminator = make_discriminator_model()

checkpoint_dir = sample_save_folder + '/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator)

seed = tf.random.normal([num_square_examples_to_generate ** 2, noise_dim])

train(train_dataset, EPOCHS)
generator.save(sample_save_folder + '/model_DCGAN.h5')