# Variational Auto-Encoder (VAE)

The goal of VAEs is to train a genrative model in the form of $p(x, z) = p(z) p(x|z)$ where $p(z)$ is a prior distribution over latent variables $z$ and $p(x|z)$ is the likelihood function or decoder that generates data $x$ given latent variables $z$. 

Since the true posterior $p(z|x)$ is in general intractable, the generative model is trained with the aid of an approximate posterior distribution or encoder $q(z|x)$.





## Hierarchical VAEs

> to increase the expressiveness

the latent variables are partitioned into disjoint groups $z = \{ z_1, z_2, \dots, z_L\}$, where $L$ is the number of groups. Then the prior is represented by $p(z) = \prod_{l} p\left(z_{l} \mid z_{<l}\right)$.







有一个mean和一个log_var



当目标分布是 $\mathcal{N}(0,1^2)$ 时，mean=0, log_var=0

当目标分布是 $\mathcal{N}(0,0.01^2)$ 时，mean=0, log_var=-9

而当mean和log_var也是分布的时候，比如初始化都是 $\mathcal{N}(0, 1^2)$

就应该分别变化到，$\mathcal{N}(0, 0.01^2)$ $\mathcal{N}(-9, 0.01^2)$ 波动小比较好
