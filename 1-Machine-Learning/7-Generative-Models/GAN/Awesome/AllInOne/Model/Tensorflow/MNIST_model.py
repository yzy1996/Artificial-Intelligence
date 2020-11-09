import tensorflow as tf
from tensorflow.keras import layers

filters_num = 64

def make_generator_model():

    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(100, )))

    model.add(layers.Dense(7 * 7 * filters_num * 2, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Reshape((7, 7, filters_num * 2)))

    model.add(layers.Conv2DTranspose(filters=filters_num * 2,
                                     kernel_size=5,
                                     strides=1,
                                     padding='same',
                                     use_bias=False,
                                     kernel_initializer=initializer))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(filters=filters_num,
                                     kernel_size=5,
                                     strides=2,
                                     padding='same',
                                     use_bias=False,
                                     kernel_initializer=initializer))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(filters=1,
                                     kernel_size=5,
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
                            kernel_size=5,
                            strides=2,
                            padding='same',
                            kernel_initializer=initializer))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(filters=filters_num * 2,
                            kernel_size=5,
                            strides=2,
                            padding='same',
                            kernel_initializer=initializer))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model
