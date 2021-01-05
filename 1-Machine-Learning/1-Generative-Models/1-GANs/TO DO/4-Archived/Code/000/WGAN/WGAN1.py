import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend
import matplotlib.pyplot as plt

# Clip model weights
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

# Calculate wasserstein loss
def wasserstein_loss(y_true, y_pred):
    return backend.mean(y_true * y_pred)

# Define Model
def generator_model():
    
    init = tf.keras.initializers.RandomNormal(stddev=0.02) # weight initialization

    model = tf.keras.Sequential()

    model.add(layers.Dense(7 * 7 * 128, kernel_initializer=init, input_dim=50))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Reshape((7, 7, 128)))
 
    # upsample to 14x14
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2,2), padding='same', kernel_initializer=init))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
 
    # upsample to 28x28
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2,2), padding='same', kernel_initializer=init))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
 
    # output 28x28x1
    model.add(layers.Conv2D(1, (7,7), activation='tanh', padding='same', kernel_initializer=init))

    return model

def discriminator_model():
    
    init = tf.keras.initializers.RandomNormal(stddev=0.02) # weight initialization
    
    const = ClipConstraint(0.01) # weight constraint

    model = tf.keras.Sequential()
 
    # downsample to 14x14
    model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init, kernel_constraint=const, input_shape=(28, 28, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
 
    # downsample to 7x7
    model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init, kernel_constraint=const))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
 
    # scoring, linear activation
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
 
    # compile model
    model.compile(loss=wasserstein_loss, optimizer=tf.keras.optimizers.RMSprop(lr=0.00005))
 
    return model

def WGAN_model(generator, discriminator):
    
    discriminator.trainable = False # make weights in the discriminator not trainable

    model = tf.keras.Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss=wasserstein_loss, optimizer=tf.keras.optimizers.RMSprop(lr=0.00005))
 
    return model

def train(generator, discriminator, WGAN_model, dataset, latent_dim=50, n_epochs=10, n_batch=64, n_critic=5):

    n_steps = int(dataset.shape[0] / n_batch) * n_epochs
    half_batch = int(n_batch / 2)

    d1_hist, d2_hist, g_hist = [], [], []

    for i in range(n_steps):
        for j in range(n_critic):
            d1_tmp, d2_tmp = [], []
            # sample real data
            X_real, y_real = dataset[np.random.randint(0, dataset.shape[0], half_batch)], -np.ones((half_batch, 1))
            # update discriminator
            d_loss1 = discriminator.train_on_batch(X_real, y_real)
            d1_tmp.append(d_loss1)

            noise = np.random.randn(latent_dim * half_batch).reshape(half_batch, latent_dim)
            X_fake, y_fake = generator.predict(noise), np.ones((half_batch, 1))
            # update discriminator
            d_loss2 = discriminator.train_on_batch(X_fake, y_fake)
            d2_tmp.append(d_loss2)
            
        d1_hist.append(np.mean(d1_tmp))
        d2_hist.append(np.mean(d2_tmp))
        X_gan, y_gan = np.random.randn(latent_dim * n_batch).reshape(n_batch, latent_dim), -np.ones((n_batch, 1))
        # # update the generator
        g_loss = WGAN_model.train_on_batch(X_gan, y_gan)
        g_hist.append(g_loss)

        print(f'iteration: {i+1} d1:{np.mean(d1_tmp):.3f}, d2:{np.mean(d2_tmp):.3f}, g:{g_loss:.3f}')



    plt.plot(d1_hist, label='disc_real')
    plt.plot(d2_hist, label='disc_fake')
    plt.plot(g_hist, label='gen')
    plt.legend()
    plt.show()


generator = generator_model()
discriminator = discriminator_model()
WGAN = WGAN_model(generator, discriminator)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images[train_labels == 7] # select a given class
train_images = np.expand_dims(train_images, axis=-1).astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]

# train model
train(generator, discriminator, WGAN, train_images)