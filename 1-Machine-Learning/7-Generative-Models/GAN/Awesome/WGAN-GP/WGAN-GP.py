import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras import layers
import time

(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5

BATCH_SIZE = 128
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)



filters_num = 128

def make_generator_model():

    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(100, )))

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
    model.add(layers.Dense(1))

    return model


def discriminator_loss_fn(real_output, fake_output):
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

def generator_loss_fn(fake_output):
    return -tf.reduce_mean(fake_output)


generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0, beta_2=0.9)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0, beta_2=0.9)

generator_loss_measure = tf.keras.metrics.Mean(name='generator_loss')
discriminator_loss_measure = tf.keras.metrics.Mean(name='discriminator_loss')


# Gradient Penalty (GP)
def gradient_penalty(discriminator, real_images, fake_images):
    t = tf.random.uniform([BATCH_SIZE, 1, 1, 1])
    inter = t * fake_images + (1 - t) * real_images
    
    with tf.GradientTape() as tape:
        tape.watch(inter)
        inter_output = discriminator(inter)
        
    gradients = tape.gradient(inter_output, inter)
    slopes = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2, 3]))
    return tf.reduce_mean((slopes - 1.) ** 2)

# 注意 `tf.function` 的使用
# 该注解使函数被“编译”
@tf.function
def train_step(images):

    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    for i in range(5):

        # update discriminator
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

    # update generator
    with tf.GradientTape() as gen_tape:

        generated_images = generator(noise, training=True)

        fake_output = discriminator(generated_images, training=True)

        generator_loss = generator_loss_fn(fake_output)
        
        generator_loss_measure.update_state(generator_loss)

    generator_gradient = gen_tape.gradient(generator_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradient, generator.trainable_variables))

def generate_and_save_images(model, epoch, test_input):
    # 注意 training` 设定为 False
    # 因此，所有层都在推理模式下运行（batchnorm）。

    sample_folder = './output3'

    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)

    predictions = model(test_input, training=False) * 127.5 + 127.5

    fig = plt.figure(figsize=(5, 5))

    for i in range(predictions.shape[0]):
        plt.subplot(5, 5, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('./output3/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        generate_and_save_images(generator, epoch + 1, seed)

        # 每 15 个 epoch 保存一次模型
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        logs = 'Epoch={}, Time={}, Generator_Loss:{}, Discriminator_Loss:{}'

        tf.print(tf.strings.format(logs, (epoch + 1 , time.time() - start, generator_loss_measure.result(), discriminator_loss_measure.result())))
    
        generator_loss_measure.reset_states()
        discriminator_loss_measure.reset_states()


generator = make_generator_model()
discriminator = make_discriminator_model()

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator)

EPOCHS = 100
noise_dim = 100
num_examples_to_generate = 25

# 我们将重复使用该种子（因此在动画 GIF 中更容易可视化进度）
seed = tf.random.normal([num_examples_to_generate, noise_dim])

train(train_dataset, EPOCHS)
generator.save('model1.h5')