import tensorflow as tf


def GAN_loss_fn():

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def discriminator_loss_fn(real_output, fake_output):
        real_loss = bce(tf.ones_like(real_output), real_output)
        fake_loss = bce(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss

    def generator_loss_fn(fake_output):
        return bce(tf.ones_like(fake_output), fake_output)

    return discriminator_loss_fn, generator_loss_fn

    # bce(tf.ones_like(fake_output), fake_output)
    # and
    # -tf.reduce_mean(tf.math.log(y_pred))
    # are same


def LSGAN_loss_fn():

    mse = tf.keras.losses.MeanSquaredError()

    def discriminator_loss_fn(real_output, fake_output):
        real_loss = mse(tf.ones_like(real_output), real_output)
        fake_loss = mse(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss

    def generator_loss_fn(fake_output):
        return mse(tf.ones_like(fake_output), fake_output)

    return discriminator_loss_fn, generator_loss_fn


def WGAN_loss_fn():

    def discriminator_loss_fn(real_output, fake_output):
        return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

    def generator_loss_fn(fake_output):
        return -tf.reduce_mean(fake_output)

    return discriminator_loss_fn, generator_loss_fn


def get_adversarial_losses_fn(mode):

    if mode == 'GAN':
        return GAN_loss_fn()

    elif mode == 'LSGAN':
        return LSGAN_loss_fn()
        
    elif mode == 'WGAN':
        return WGAN_loss_fn()


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

