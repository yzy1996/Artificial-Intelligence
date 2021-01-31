<h1 align="center">Implement of GANs</h1>
<div align="center">

Latest implement of various GANs with tensorflow(=2.3.0) and pytorch(=1.16.0).

More details about GANs, please see the repository.

You can also get access to the code in Google Colab.

</div>



## Folder Structure

The following shows basic folder structure.

```
├── main.py # gateway
├── data
│   ├── mnist # mnist data (not included in this repo)
│   ├── ...
│   ├── ...
│   └── fashion-mnist # fashion-mnist data (not included in this repo)
│
├── GAN.py # vainilla GAN
├── utils.py # utils
├── dataloader.py # dataloader
├── models # model files to be saved here
└── results # generation results to be saved here
```



## Dataset

- MNIST
- CelebA
- Fashion-MNIST
- CIFAR10



## List



| Name (paper link) |                        Loss Function                         |
| :---------------: | :----------------------------------------------------------: |
|  **Vanilla GAN**  |                    $$L_D^{GAN} = E\\as$$                     |
|     **LSGAN**     |                                                              |
|     **WGAN**      | $$\begin{align}&L_{D}^{WGAN}=-\mathbb{E}_{x \sim p_{\text {data }}}[D(x)]+\mathbb{E}_{z \sim p(z)}[D(G(z))]\\&L_{G}^{WGAN}=-\mathbb{E}_{z \sim p(z)}[D(G(z))]\\&W_D \leftarrow \text{clip_by_value}(W_D, -0.01, 0.01) \end{align}$$ |
|    **WGAN-GP**    | $$L_{D}^{WGAN-GP}=L_{D}^{WGAN} + \lambda \mathbb{E}[(|\nabla_xD(x)|-1)^2 ]\\L_{G}^{WGAN-GP}=L_{G}^{WGAN}$$ |
|                   |                                                              |

> inspired by [hwalsuklee](https://github.com/hwalsuklee/tensorflow-generative-model-collections)



### Vanilla GAN

[Generative Adversarial Networks](https://arxiv.org/pdf/1406.2661.pdf)

**[`NeurIPS 2014`]**

**[`Ian J. Goodfellow`, `Jean Pouget-Abadie`, `Mehdi Mirza`, `Bing Xu`, `David Warde-Farley`, `Sherjil Ozair`, `Aaron Courville`, `Yoshua Bengio`]**

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

equal to

```python
def discriminator_loss_fn(real_output, fake_output):
	return -tf.reduce_mean(tf.math.log(real_output + 1e-10) + tf.math.log(1. - fake_output + 1e-10))

def generator_loss_fn(fake_output):
	return -tf.reduce_mean(tf.math.log(fake_output + 1e-10))
```



### DCGAN

[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf)

**`[ICLR 2016]`**

**[`Alec Radford`, `Luke Metz`, `Soumith Chintala`]**

**Main contributions**

- Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).
- Use Batchnorm in both the generator and the discriminator
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



### LSGAN

[Least Squares Generative Adversarial Networks](https://arxiv.org/pdf/1611.04076.pdf)

**[`ICCV 2017`]**

**[`Xudong Mao`, `Qing Li`, `Haoran Xie`, `Raymond Y.K. Lau`, `Zhen Wang`, `Stephen Paul Smolley`]**

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



### WGAN

> [click to learn more]()

[Wasserstein GAN](https://arxiv.org/abs/1701.07875)

**[`ICML 2017`]**

**[`Martin Arjovsky`, `Soumith Chintala`, `Léon Bottou`]**

**[[code](https://github.com/martinarjovsky/WassersteinGAN)]** 

**Main contributions**

- No log in the loss. Use a linear activation function in the output layer of the D (instead of sigmoid).

- Use -1 labels for real images and 1 labels for fake images (instead of 1 and 0).

- Use Wasserstein loss to train the critic and generator models.

- Clip the weights of D to a limited range after each mini batch update (e.g. [-0.01,0.01]).

- Train D more than G each iteration (e.g. 5).

- Use the RMSProp version of gradient descent with a small learning rate and no momentum (e.g. 0.00005).

**Loss function**

![mylatex20201106_204405](https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20201106204634.png)

**Loss function code**

```python
def discriminator_loss_fn(real_output, fake_output):
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

def generator_loss_fn(fake_output):
    return -tf.reduce_mean(fake_output)
```

等价于

```python
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss_fn(real_output, fake_output):
    real_loss = bce(-tf.ones_like(real_output), real_output)
    fake_loss = bce(tf.ones_like(fake_output), fake_output)
    return real_loss - fake_loss

def generator_loss_fn(fake_output):
    return bce(-tf.ones_like(fake_output), fake_output)
```



### WGAN-GP

> [click to learn more]()

[Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)

**[`NeurIPS 2017`]**

**[`Ishaan Gulrajani`, `Faruk Ahmed`, `Martin Arjovsky`, `Vincent Dumoulin`, `Aaron Courville`]**

**[[code](https://github.com/igul222/improved_wgan_training)]** 

**Loss function**

![mylatex20201106_212247](https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20201106212304.png)

**Loss function code**

```python
def discriminator_loss_fn(real_output, fake_output):
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

def generator_loss_fn(fake_output):
    return -tf.reduce_mean(fake_output)

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

discriminator_loss_fn() += 10 * gradient_penalty()
```






## Acknowledgements

based on [tensorflow-generative-model-collections](https://github.com/hwalsuklee/tensorflow-generative-model-collections) and [pytorch-generative-model-collections](https://github.com/znxlwm/pytorch-generative-model-collections). 