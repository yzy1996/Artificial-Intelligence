'''

'''

import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from pathlib import Path
import time


# 定义网络模型
def make_generator_model():

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(10, )))

    model.add(layers.Dense(100, activation='relu'))

    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(50, activation='relu'))

    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(2))

    return model

generator = make_generator_model()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def loss_fn(generated_solve):
    out = generated_solve[:, 0]**2 + generated_solve[:, 1]**2 - 1
    return cross_entropy(tf.zeros_like(out), out)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)

generator_loss_measure = tf.keras.metrics.Mean(name='generator_loss')

@tf.function
def train_step():

    noise = tf.random.normal([128, 10])

    with tf.GradientTape() as gen_tape:

        generated_solve = generator(noise, training=True)

        generator_loss = loss_fn(generated_solve) ** 2

        generator_loss_measure.update_state(generator_loss)

    generator_gradient = gen_tape.gradient(generator_loss, generator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradient, generator.trainable_variables))



def train(epochs):

    for epoch in range(epochs):

        start = time.time()

        train_step()

        prediction = generator(seed, training=False)

        print(prediction)

        tf.print(f'[{epoch + 1}/{EPOCHS}], loss = {generator_loss_measure.result():.3f}, time = {time.time() - start:.3f}')
      
        generator_loss_measure.reset_states()




seed = tf.random.normal([NUM_TO_GENERATE, NOISE_DIM])

train(50)




# x_train = 