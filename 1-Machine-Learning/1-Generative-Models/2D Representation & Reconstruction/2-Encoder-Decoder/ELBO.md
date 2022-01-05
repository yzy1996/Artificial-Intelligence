# ELBO

> Evidence Lower Bound

首先是问题的设定：针对带有隐变量的概率模型

我们有随机变量 $X, Z$，他们服从一个联合分布 $p(X,Z;\theta)$，我们的数据只有对 $X$ 实现的观测，对 $Z$ 是不知道的。因此一般我们有两个任务想要实现：

- 给定 $\theta$ ，计算后验分布 $p(Z|X;\theta)$
- 用最大似然估计 $\theta$，$\arg \max_\theta \{ \log p(x;\theta) = \log \int_z p(x,z;\theta) dz\}$



变分推断就是用在任务1上。



Evidence (证据) 定义的就是 似然函数 $\log p(x;\theta)$。之所以被称为 证据，是因为它能反应我们对模型的估计好坏程度。如果我们知道了 $Z$ 服从的分布 $q$，(在后面的表示中，为了简化我们省去了 $\theta$)


$$
\begin{align}
\log p(x) 
&= \log \int p(x,z) dz \\
&= \log \int p(x,z) \frac{q(z)}{q(z)} dz \\
&= \log \mathbb{E}_{z \sim \mathcal{Z}} \left[ \frac{p(x, z)}{q(z)} \right] \\
&\ge \mathbb{E}_{z \sim \mathcal{Z}} \log \left[ \frac{p(x, z)}{q(z)} \right]
\end{align}
$$

$$
ELBO := \mathbb{E}_{z \sim \mathcal{Z}} \log \left[ \frac{p(x, z)}{q(z)} \right]
$$


The gap between the evidence and the ELBO is the Kullback-Leibler Divergence between $p(z|x)$ and $q(z)$:
$$
\begin{aligned}
KL(q(z) \| p(z \mid x)) 
&:= \mathbb{E}_{z \sim \mathcal{Z}} \log \left[ \frac{q(z)}{p(z \mid x)} \right] \\
&= \mathbb{E}_{z \sim \mathcal{Z}} \left[\log q(z)\right] - \mathbb{E}_{z \sim \mathcal{Z}} \left[\log p(x,z)\right] + \mathbb{E}_{z \sim \mathcal{Z}} \left[\log p(x)\right] \\
&= \log p(x) - \mathbb{E}_{z \sim \mathcal{Z}} \log \left[ \frac{p(x, z)}{q(z)} \right] \\
&= \text{Evidence} - \text{ELBO}
\end{aligned}
$$




Background: 

variable 

latent variables





We want to use a $q(z)$ approximate $p(z|x)$ and get an optimal $q^*(z)$


$$
\text{KL }(q(z) \| p(z \mid x)) = \mathbb{E}[\log q (z)] - \mathbb{E}[\log p (z, x)] + \log p(x)
$$
add 
$$
KL() \ge 0
$$
so

$$
ELBO(q) = \mathbb{E}[\log p (z, x)] - \mathbb{E}[\log q (z)]
$$





吉布斯不等式 

若 $\sum_{i=1}^n p_i = \sum_{i=1}^n p_i = 1$，且 $p_i, q_i \in (0, 1]$，则有：
$$
-\sum_{i=1}^{n} p_{i} \log p_{i} \leq-\sum_{i=1}^{n} p_{i} \log q_{i} \text { ，等号成立当且仅当 } p_{i}=q_{i} \ \forall i
$$


最小化KL散度等价于最大化ELBO



杰森不等式 Jensen's Inequality







如果我们从后验的角度来看，首先什么是后验，

在生成模型中，我们是从一个z得到一个x，后验概率就是 p(z|x)


$$
P(Z \mid X)=\frac{p(X, Z)}{\int_{z} p(X, Z=z) d z}
$$
所以变分推断 (Variational Inference)，是为了 推断 z 



因此我们想引入一个参数化模型 p(z;\lambda) 来近似 p(z|x)，相当于是用一个简单分布去拟合了一个复杂分布
$$
\lambda^{*}=\arg \min _{\lambda} \text { divergence }(p(z \mid x), q(z ; \lambda))
$$
这个度量一般就是用KL散度，
$$
D_{K L}(p \| q)=\sum_{i=1}^{N}\left[p\left(x_{i}\right) \log p\left(x_{i}\right)-p\left(x_{i}\right) \log q\left(x_{i}\right)\right] = \sum_{i=1}^{N} p\left(x_{i}\right) \log \left(\frac{p\left(x_{i}\right)}{q\left(x_{i}\right)}\right)
$$

> 因为KL散度大于0，很好用梯度下降



所以我们的目标是最小化 $\min _{\lambda} K L(q(z ; \lambda) \| p(z \mid x))$

直接求很难，让我们回到贝叶斯公式


$$
p(x) = \frac{p(x,z)}{p(z|x)}
$$
两边取log，
$$
\begin{aligned}
\log P(x) &=\log P(x, z)-\log P(z \mid x) \\
&=\log \frac{P(x, z)}{Q(z ; \lambda)}-\log \frac{P(z \mid x)}{Q(z ; \lambda)}
\end{aligned}
$$
然后再对q求期望
$$
\begin{aligned}
\mathbb{E}_{q(z ; \lambda)} \log P(x) &=\mathbb{E}_{q(z ; \lambda)} \log P(x, z)-\mathbb{E}_{q(z ; \lambda)} \log P(z \mid x) \\
\log P(x) &=\mathbb{E}_{q(z ; \lambda)} \log \frac{p(x, z)}{q(z ; \lambda)}-\mathbb{E}_{q(z ; \lambda)} \log \frac{p(z \mid x)}{q(z ; \lambda)} \\
&=K L(q(z ; \lambda) \| p(z \mid x))+\mathbb{E}_{q(z ; \lambda)} \log \frac{p(x, z)}{q(z ; \lambda)}
\end{aligned}
$$

$$
\max _{\lambda} \mathbb{E}_{q(z ; \lambda)} \log \frac{p(x, z)}{q(z ; \lambda)}
$$

$$
\log P(x)=K L(q(z ; \lambda) \| p(z \mid x))+E L B O
$$
Any procedure which uses optimization to approximate a density can be termed ``variational inference''.

Jordan (2008) 曾经对 Variational Inference 给出来一个直观的定义：



Variational Bayes is a particular variational method which aims to find some approximate joint distribution Q(x; θ) over hidden variables x to approximate the true joint P(x), and defines ‘closeness’ as the KL divergence KL[Q(x; θ)||P(x)].



概率模型中的后验分布推断常见的方法是：MAP、EM算法、变分推断(Variational Inference, VI)和蒙特卡洛推断(Monte Carlo Inference, MCI)。可以粗暴地、不严谨地理解为，EM是VI的特例，VI是MCI的特例。


$$
\log p(x)=E L B O+K L D
$$




先说结论，
$$
\text{ELBO} = E_{q} \log p(\theta, \beta, z, w \mid \alpha, \eta)-E_{q} \log q(\beta, z, \theta \mid \lambda, \phi, \gamma)
$$
