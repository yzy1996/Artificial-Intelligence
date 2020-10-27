# DCGAN

[`Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks`]
[`ICLR`] [`2016`]


$$
L_D = E[\log(D(x))] + E[\log(1-D(G(z)))]
$$

$$
L_G = E[\log(D(G(z)))]
$$



判别器给 真数据 



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



## code



0 for fake output

1 for real output 