# Generative model









## Background

photorealistic image synthesis

- high resolution
- content controllable



2D\3D



compositional nature of scenes

- individual objects' shapes
- appearances
- background



Modern computer graphics (CG) techniques have achieved impressive results and are industry standard in gaming and movie productions. However, they are very hardware and computing expensive and require substantial repetitive labor. 

Therefore, the ability to generate and manipulate photorealistic image content is a long-standing goal of computer vision and graphics.











## Introduction

Generative models can be divided into two classes:

- implicit generative models (IGMs)
- explicit generative models (EGMs)



Our goal is to train a model $\mathbb{Q}_{\theta}$ which aims to approximate a target distribution $\mathbb{P}$ over a space $\mathcal{X} \subseteq \mathbb{R}^{d}$.

Normally we define $\mathbb{Q}_{\theta}$ by a generator function $G_{\theta}: \mathcal{Z} \rightarrow \mathcal{X}$, implemented as a deep network with parameters $\theta$, where $\mathcal{Z}$ is a space of latent vectors, say $\mathcal{R}^{128}$. We assume a fixed Gaussian distribution on $\mathcal{Z}$, and call $\mathbb{Q}_{\theta}$ the distribution of $G_{\theta}(Z)$. 

The optimization process is to learn by minimizing a discrepancy $\mathcal{D}$ between distributions , with the property $\mathcal{D}(\mathbb{P}, \mathbb{Q}_{\theta}) \geq 0$ and $\mathcal{D}(\mathbb{P}, \mathbb{P})=0$.



we can build loss $\mathcal{D}$ based on the Maximum Mean Discrepancy,
$$
\operatorname{MMD}_{k}(\mathbb{P}, \mathbb{Q})=\sup _{f:\|f\|_{\mathcal{H}_{k}} \leq 1} \mathbb{E}_{X \sim \mathbb{P}}[f(X)]-\mathbb{E}_{Y \sim \mathbb{Q}}[f(Y)]
$$
where $\mathcal{H}_k$ is the reproducing kernel Hilbert space with a kernel $k$.





Wasserstein distance
$$
\mathcal{W}(\mathbb{P}, \mathbb{Q})=\sup _{f:\|f\|_{\text {Lip }} \leq 1} \mathbb{E}_{X \sim \mathbb{P}}[f(X)]-\mathbb{E}_{Y \sim \mathbb{Q}}[f(Y)]
$$





There are three main methods: 

- VAE

- GAN
- Flow

They both learn from the training data and use the learned model to generate or predict new instances.



相同点：都用到了随机噪声，然后度量噪声和真实数据的分布差异

不同点：GAN为了拟合数据分布，VAE为了找到数据的隐式表达，Flow建立训练数据和生成数据之间的关系

GAN 和 Flow 的输入和输出都是一一对应的，而VAE不是



训练的损失函数上：

VAE最大化ELBO，其目的是要做最大似然估计，最大似然估计等价于最小化KL，但这个KL不是数据和噪声的KL，而是model给出的![[公式]](https://www.zhihu.com/equation?tex=p%28x%29)和数据所展示的![[公式]](https://www.zhihu.com/equation?tex=p%28x%29)之间的KL。

GAN是最小化JS，这个JS也是model给出的![[公式]](https://www.zhihu.com/equation?tex=p%28x%29)和数据所展示的![[公式]](https://www.zhihu.com/equation?tex=p%28x%29)之间的。

流模型训练也非常直接，也是最大似然估计。只不过因为流模型用的是可逆神经网络，因此，相比于其他两者，学习inference即学习隐含表示非常容易，




## GAN 2014

Generative Adversarial Networks (GANs) emerge as a powerful class of generative models. In particular, they are able to synthesize photorealistic images at high resolutions ($$1024 \times 1024$$) pixels which can not be distinguished. 



GANs and its variants 



train with adversarial methods, bypass the need of computing densities, at the expense of a good density estimation

Generative adversarial networks (GANs) represent a zero-sum game between two machine players, a generator and a discriminator, designed to learn the distribution of data.



> 只要能骗过Discriminator就好



## VAE 2013

at the cost of learning two neural networks





## VAE-GAN

combine VAE with GAN



## Bijective GNN



## Flow



