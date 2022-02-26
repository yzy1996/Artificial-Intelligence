### 一文解读VAE

> Variational Auto-Encoder，来自论文 
> 
> [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf)  
> *Diederik P Kingma, Max Welling*  
> **[`ICLR 2014`] (`Universiteit van Amsterdam`)**



像GAN一样，VAE也是想要学习一个生成模型以实现一个从低维 latent vector $z \in \mathbb{R}^d$ 到高维 $x \in \mathbb{D}$ 的映射 $g: \mathcal{Z} \rightarrow \mathcal{X}$。VAE的优化目标是最大化数据集数据的似然：
$$
\log p(X) = \sum_{i=1}^N \log \sim
$$








首先我们有一批数据样本 $\mathbf{x}= \{x_1, x_2, \dots, x_n\}$，现要估计它的分布 $p(x)$。

我们要得到的结果是 $p_\theta (x|z)$ 然后都在讲 怎么推断 后验分布 p(z|x)

我们想借助隐变量 $z$ 来描述 $\mathbf{x}$ 的分布，建模成：
$$
q(x)=\int q(x, z) d z, \quad q(x, z)=q(x \mid z) q(z)
$$
$x$ 和 $z$ 的联合分布还可以写成 $p(x,z) = p(z|x) p(x)$。因此我们想用 $q(x,z)$ 来近似 $p(x,z)$。因此直接用KL散度来衡量（KL散度越小越好）：
$$
\begin{aligned}
K L(p(x, z) \| q(x, z)) &=
\iint p(x,z) \ln \frac{p(x,z)}{q(x,z)} dzdx\\
&=\int p(x)\left[\int p(z \mid x) \ln \frac{p(x) p(z \mid x)}{q(x, z)} d z\right] d x \\
&=\mathbb{E}_{x \sim p(x)}\left[\int p(z \mid x) \ln \frac{p(x) p(z \mid x)}{q(x, z)} d z\right]\\
&=\mathbb{E}_{x \sim p(x)}\left[\int p(z \mid x) \left(\ln p(x) + \ln \frac{p(z \mid x)}{q(x,z)} \right)dz\right]\\
&=\mathbb{E}_{x \sim p(x)}\left[\int p(z \mid x) \ln p(x) dz\right] + \mathbb{E}_{x \sim p(x)}\left[\int p(z \mid x)\ln \frac{p(z \mid x)}{q(x,z)} dz\right]\\
&=\mathbb{E}_{x \sim p(x)}\left[\ln p(x) \int p(z \mid x) dz\right] + \mathbb{E}_{x \sim p(x)}\left[\int p(z \mid x)\ln \frac{p(z \mid x)}{q(x,z)} dz\right]\\
&=\mathbb{E}_{x \sim p(x)}\left[\ln p(x) \right] + \mathbb{E}_{x \sim p(x)}\left[\int p(z \mid x)\ln \frac{p(z \mid x)}{q(x,z)} dz\right]\\
\end{aligned}
$$
注意第一项可以看成是一个常数。因此我们可以将求KL散度的问题转化为一个新的损失函数为：
$$
\begin{aligned}
\mathcal{L} 
&= \mathbb{E}_{x \sim p(x)}\left[\int p(z \mid x)\ln \frac{p(z \mid x)}{q(x,z)} dz\right]\\
&= \mathbb{E}_{x \sim p(x)}\left[\int p(z \mid x)\ln \frac{p(z \mid x)}{q(x \mid z)q(z)} dz\right]\\
&= \mathbb{E}_{x \sim p(x)}\left[-\int p(z \mid x)\ln q(x \mid z)dz + \int p(z \mid x) \ln \frac{p(z \mid x)}{q(z)} dz\right]\\
&= \mathbb{E}_{x \sim p(x)}\left[\mathbb{E}_{z \sim p(z \mid x)}[-\ln q(x \mid z)]+KL\left(p(z \mid x) \| q(z)\right)\right]
\end{aligned}
$$
最终目的就是优化 $q(x \mid z), q(z)$ 让 $\mathcal{L}$ 最小。



（先休息一下）



现在我们有 $q(z), q(x|z), p(z|x)$ 是未知的，因此实验中我们要确定他们的形式。

- $q(z)$：我们直接假设 $z \sim N(0, I)$
- $p(z|x)$：也假设是正态分布，均值和方差是可学习的参数。
- $q(x|z)$： 也假设是正态分布，均值和方差是可学习的参数。



因为要计算 $\mathbb{E}_{z \sim p(z \mid x)}[-\ln q(x \mid z)]$，就需要对 $z \sim p(z|x)$ 进行采样，VAE论文说只需要每次采样一个就够了，每个循环都是随机的，因此采样是足够充分的。所以最终 $\mathcal{L}$ 变成了：
$$
\mathcal{L}=\mathbb{E}_{x \sim p(x)}[-\ln q(x \mid z)+K L(p(z \mid x) \| q(z))], \quad z \sim p(z \mid x)
$$
MSE, 


$$
\begin{gathered}
D_{K L}(q(Z \mid X) \| p(Z \mid X))=\mathbb{E}[\log (q(Z \mid X))-\log (p(Z \mid X))] \\
=\mathbb{E}\left[\log (q(Z \mid X))-\log \left(\frac{p(X \mid Z) p(Z)}{p(X)}\right)\right] \\
=\mathbb{E}[\log (q(Z \mid X))-\log (p(X \mid Z)-\log (p(Z)))]+\log (p(X)) \\
=\mathbb{E}\left[\log \left(\frac{q(Z \mid X)}{p(Z)}\right)-\log (p(X \mid Z))\right]+\log (p(X)) \\
=D_{K L}[q(Z \mid X)|| p(Z)]-\mathbb{E}[\log (p(X \mid Z))]+\log (p(X))
\end{gathered}
$$
等号右边第一项不就是似然值吗？第二项只要实现把先验概率 ![[公式]](https://www.zhihu.com/equation?tex=p%28Z%29) 定义好之后，也可以进行计算。



因为 $p(x|z)$ 形如 decoder，而 $q(z|x)$ 形如 Encoder，因此得名 VAE。和 Auto-Encoder 并没有那么大的关系







重参数化的作用

如果直接从多元正态分布去采样，破坏了连续性，



![img](https://pic1.zhimg.com/80/v2-f60be7abe507be3c176135d875864280_1440w.jpg?source=1940ef5c)





## VAE++

提到



我们可以统一VAE以及变种的模型，第一项保证重建的准确率高，第二项保证编码到的latent 分布和先验的是一致的。
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









Auto-encoding variational bayes

Multi-level variational autoencoder: Learning disentangled representations from grouped observations

Extracting and composing robust features with denoising autoencoders

Semantic facial expression editing using autoencoded flow

betavae: Learning basic visual concepts with a constrained variational framework

















## 参考链接

[苏剑林-变分自编码器（一）：原来是这么一回事](https://kexue.fm/archives/5253)

[苏剑林-变分自编码器（二）：从贝叶斯观点出发](https://kexue.fm/archives/5343)
