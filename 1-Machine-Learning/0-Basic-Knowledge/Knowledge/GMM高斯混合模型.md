# GMM高斯混合模型

> Gaussian Mixed Model

设有随机变量$X$，则混合高斯模型可以用下式表示：
$$
p(\boldsymbol{x})=\sum_{k=1}^{K} \pi_{k} \mathcal{N}\left(\boldsymbol{x} | \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)
$$
其中 $\mathcal{N}\left(\boldsymbol{x} | \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)$ 成为混合模型中的第k个分量（component），$\pi_k$ 是混合系数，它满足条件 $\sum_{k=1}^{K} \pi_{k}=1$ ，$0 \leq \pi_{k} \leq 1$ 



我们引入一个新的K维随机变量z，$z_k$ 只能取0或1两个值，$z_k=1$ 表示第k类被选中的概率，即 $p(z_k=1)=\pi_k$ ；$z_k$ 要满足两个条件： $z_{k} \in\{0,1\}$，$\sum_{K} z_{k}=1$

z的联合概率分布可以表示为：

$$
p(\boldsymbol{z})=p\left(z_{1}\right) p\left(z_{2}\right) \ldots p\left(z_{K}\right)=\prod_{k=1}^{K} \pi_{k}^{z_k}
$$

第k类中的数据服从高斯分布，
$$
p(\boldsymbol{x} | \boldsymbol{z})=\prod_{k=1}^{K} \mathcal{N}\left(\boldsymbol{x} | \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)^{z_k}
$$

根据贝叶斯公式，可以得到
$$
\begin{aligned} 
p(\boldsymbol{x}) &=\sum_{\boldsymbol{z}} p(\boldsymbol{z}) p(\boldsymbol{x} | \boldsymbol{z}) \\ &=\sum_{\boldsymbol{z}}\left(\prod_{k=1}^{K} \pi_{k}^{z_{k}} \mathcal{N}\left(\boldsymbol{x} | \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)^{z_{k}}\right) \\ 
&=\sum_{k=1}^{K} \pi_{k} \mathcal{N}\left(\boldsymbol{x} | \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right) 
\end{aligned}
$$
隐变量 $z$ 的含义是，对于一个数据点，我们不知道它属于哪一类，也就是不知道它属于哪一个高斯分布分量。

后验概率
$$
\gamma(z_{k})=p(z|x)=\frac{\pi_{k} \mathcal{N}\left(\boldsymbol{x} | \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)}{\sum_{k=1}^{K} \pi_{k} \mathcal{N}\left(\boldsymbol{x} | \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)}
$$
用 $\gamma(z_k)$ 来表示第k个分量的后验概率



GMM模型
$$
p(\boldsymbol{x} | \boldsymbol{\pi}, \boldsymbol{\mu}, \boldsymbol{\Sigma})=\sum_{k=1}^{K} \pi_{k} \mathcal{N}\left(\boldsymbol{x} | \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)
$$
所有样本连乘得到似然函数：
$$
l=log[\prod_{i=1}^{n}p(x_i|\pi,\mu,\Sigma)]=\sum_{i=1}^{n}log[\sum_{k=1}^{K}\pi_{k} \mathcal{N}\left({x_i} | \mu_{k}, \Sigma_{k}\right)]
$$
对 $\mu_k$ 求导并令导数为0
$$
\frac{\partial{l}}{\partial{\mu_k}}=\sum_{i=1}^{n}\frac{}{\sum_{k=1}^{K}\pi_{k} \mathcal{N}\left({x_i} | \mu_{k}, \Sigma_{k}\right)}
$$







