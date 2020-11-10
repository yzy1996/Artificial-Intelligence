import tensorflow as tf

# WGAN

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