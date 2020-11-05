# DCGAN

[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)

Alec Radford, Luke Metz, Soumith Chintala **`[ICLR 2016]`**



**Architecture guidelines for stable Deep Convolutional GANs**

- Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).
- Use batchnorm in both the generator and the discriminator
- Remove fully connected hidden layers for deeper architectures. Just use average pooling at the end.
- Use ReLU activation in generator for all layers except for the output, which uses Tanh.
- Use LeakyReLU activation in the discriminator for all layers.



**Loss function**

![mylatex20201105_204317](https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20201105204330.png)

**Loss function code**

```python
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss_fn(real_output, fake_output):
    real_loss = bce(tf.ones_like(real_output), real_output)
    fake_loss = bce(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss_fn(fake_output):
    return bce(tf.ones_like(fake_output), fake_output)
```

