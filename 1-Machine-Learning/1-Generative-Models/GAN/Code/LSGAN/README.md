# LSGAN

[Least Squares Generative Adversarial Networks](https://arxiv.org/abs/1611.04076)

Xudong Mao, Qing Li, Haoran Xie, Raymond Y.K. Lau, Zhen Wang, Stephen Paul Smolley **`[ICCV 2017]`**





**Loss function**

![mylatex20201106_203527](https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20201106203618.png)

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


