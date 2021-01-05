# Generative model

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








## GAN 2014

train with adversarial methods, bypass the need of computing densities, at the expense of a good density estimation

Generative adversarial networks (GANs) represent a zero-sum game between two machine players, a generator and a discriminator, designed to learn the distribution of data.



## VAE 2013

at the cost of learning two neural networks



## Bijective GNN

