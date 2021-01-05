# Vanilla GAN

[Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)

Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio **`[NeurIPS 2014]`**



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

等价于

```python
def discriminator_loss_fn(real_output, fake_output):
	return -tf.reduce_mean(tf.math.log(real_output + 1e-10) + tf.math.log(1. - fake_output + 1e-10))

def generator_loss_fn(fake_output):
	return -tf.reduce_mean(tf.math.log(fake_output + 1e-10))
```

