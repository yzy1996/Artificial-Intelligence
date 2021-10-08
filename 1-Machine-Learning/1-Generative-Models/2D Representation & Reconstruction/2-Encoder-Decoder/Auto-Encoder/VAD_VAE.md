# Variational Auto-Encoder (VAE)



KLD = 



有一个mean和一个log_var



当目标分布是 $\mathcal{N}(0,1^2)$ 时，mean=0, log_var=0

当目标分布是 $\mathcal{N}(0,0.01^2)$ 时，mean=0, log_var=-9

而当mean和log_var也是分布的时候，比如初始化都是 $\mathcal{N}(0, 1^2)$

就应该分别变化到，$\mathcal{N}(0, 0.01^2)$ $\mathcal{N}(-9, 0.01^2)$ 波动小比较好，



DeepSDF





本来是要优化
$$
\max \frac{1}{N} \log p(X) = \frac{1}{N} \sum^N_{i=1} \log \int p\left(x_{i}, z\right) d z
$$
但是这个问题太难优化了，所以转而优化一个下界
$$
\max \log p(x_i) \ge L(x_i) = \mathbb{E} [-\log q_\phi (z \mid x_i) + \log p_\theta (x_i, z)]
$$



问题建模是：


$$
\mathbf{z} \sim q_\phi(\mathbf{z}|\mathbf{x})
$$




采样z，$p(z)$ 是一个 prior distribution over the latent space
$$
\mathbf{z} \sim p(\mathbf{z})
$$
z 经过 decoder 到 x，参数为 $\theta$
$$
\mathbf{x} \sim p_\theta(\mathbf{x}|\mathbf{z})
$$


posterior $\mathbf{z} \sim q_\phi(\mathbf{z}|\mathbf{x})$

true posterior 

Variational Inference is used to approximate the posterior by minimizing the Kullback-Leibler (KL)-divergence between the approximate posterior and the true posterior



第一项保证重建的准确率高，第二项保证编码到的latent 分布和先验的是一致的。







理论上和实际上是不一样的，理论上是ELBO

实际上呢


$$
L_{\text{VAE}} (\theta, \phi) = L_{\text{recons}}(\theta, \phi) + L_{\text{KLD}}(\theta, \phi)
$$


The reconstruction loss is:
$$
L_{\text{recons}}(\theta, \phi) = \frac{1}{N} \sum_{i=1}^N \| \hat{\mathbf{x}_i} - \mathbf{x}_i \|_2^2
$$
The regularization loss is:
$$
L_{\text{KLD}}(\theta, \phi) = D_{\text{KL}} \left(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}) \right)
$$
another $\beta$-VAE uses the following formulation:
$$
L_{\text{VAE}} (\theta, \phi) = L_{\text{recons}}(\theta, \phi) + \beta L_{\text{KLD}}(\theta, \phi)
$$
$\beta > 1$ would encourages the independence of the dimensions of the latent space and leads to better disentanglement.



