# LSGAN

[`Least Squares Generative Adversarial Networks`]
[`ICCV`] [`2017`]



$$
L_D = E[(D(x) - 1)^2] + E[D(G(z))^2]
$$

$$
L_G = E[(D(G(z)) - 1)^2]
$$



**Loss function code**

```python
mse = tf.keras.losses.MeanSquaredError()

def discriminator_loss_fn(real_output, fake_output):
    real_loss = mse(tf.ones_like(real_output), real_output)
    fake_loss = mse(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss_fn(fake_output):
	return mse(tf.ones_like(fake_output), fake_output)
```



## code



0 for fake output

1 for real output 