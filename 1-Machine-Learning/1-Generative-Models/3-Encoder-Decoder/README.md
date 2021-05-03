# 对比 AutoEncoder 和 AutoDecoder



## Introduction

一些概述性的话，



[Optimizing the Latent Space of Generative Networks 2018]()

文中提到了在训练GAN的时候，同时优化 $G$ 的参数以及每张图片的潜在变量 $z_i$ 。

> we jointly learn the parameters $\theta$ in $\Theta$ of a generator $g_{\theta}$ and the optimal noise vector $z_i$ for each image $x_i$, by solving: ($\ell$ is a loss function)
> $$
> \min _{\theta \in \Theta} \frac{1}{N} \sum_{i=1}^N\left[\min _{z_{i} \in \mathcal{Z}} \ell\left(g_{\theta}\left(z_{i}\right), x_{i}\right)\right]
> $$

Autoencoder 是用一个参数化的模型 $f:\mathcal{X} \mapsto \mathcal{Z}$，然后最小化重建loss $\ell(g(f(x)),x)$。而上述过程是无参数的，不仅可以包含AE能找到的所有解的，也可以找到更新的解。

文中命名上，称为“encoder-less autoencoder” 或者 “discriminator-less GAN”









首先分别介绍两者结构，以下将用AE和AD分别指代全称

## AutoEncoder

![AutoEncoder](https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210329100645.svg)



论文：

Reducing the Dimensionality of Data with Neural Networks *Hinton et al*.



训练过程是：

推断过程是：

采样过程？是随机像GAN那样采样还是从已经训练过的数据得到的z里面去组合采样



任务目标：

- 降维 dimension reduction

这个其实很好理解，中间的 bottleneck 隐变量 就是去掉了不重要的，留下最重要的表征，有点像PCA或者MF (Matrix Factorization)。

- 生成

## AutoDecoder

![AutoDecoder](https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210329100810.svg)

训练过程是：

推断过程是：



## 优劣比较

AE是欠拟合的





- AD可以做到增量学习

知识库可以做到更新，训练数据不固定





> 思考？我们人的认知过程是AE还是AD呢？



黑夜中看到一个物体的棱角，我们是不会去想办法表征的，而是会去猜测，然后一步步走进加强（改进）这个猜测，而这里能够改变的就是z，而不是Decoder



