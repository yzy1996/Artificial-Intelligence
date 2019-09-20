# 概率相关知识



## 概率函数

概率质量函数：Probability mass function (PMF)，分布律

概率密度函数：Probability density function (PDF)

概率质量函数：Cumulative distribution function (CDF)



**PMF**是针对离散随机变量而言的，是随机变量在各特定取值上的概率

> 例如抛骰子，每一面朝上的概率都是1/6，那么表示出的PMF就是：
>
> $f_{X}(x)=\left\{\begin{array}{l}{\frac{1}{6} \text { if } x \in\{1,2,3,4,5,6\}} \\ {0 \text { if } x \notin\{1,2,3,4,5,6\}}\end{array}\right.$



**PDF**是针对连续随机变量而言的，因为连续所以我们无法描述变量落在某一点上的概率，只能说落在某一区间上的概率，官方描述为：在某个确定的取值点附近的可能性的函数

我们经常看到的均匀分布，高斯分布，说的就是概率密度函数。

> 例如均匀分布，它的概率密度函数是：
>
> $f(x)=\left\{\begin{array}{ll}{\frac{1}{b-a}} & {\text { for } a \leq x \leq b} \\ {0} & {\text { elsewhere }}\end{array}\right.$
>
> ![1567845043390](https://img-blog.csdnimg.cn/20190920095635829.png)
>
> 如果我们说落在每一点上的概率是$\frac{1}{b-a}$，那么岂不是（b-a）个点就使得总概率为1了吗？所以并不是这样，应该是PDF函数的积分才是函数，也即PDF函数图像的面积。



## 二项分布、泊松分布、正态分布关系

1. 泊松分布，二项分布都是离散分布；正态分布是连续分布。

2. 二项分布什么时候趋近于泊松分布，什么时候趋近于正态分布？

   这么说吧：二项分布有两个参数，一个 n 表示试验次数，一个 p 表示一次试验成功概率。
   现在考虑一个二项分布，其中试验次数 n 无限增加，而 p 是 n 的函数。
   如果 np 存在有限极限 λ，则这列二项分布就趋于参数为 λ 的 泊松分布。反之，如果 np 趋于无限大（如 p 是一个定值），则根据德莫佛-拉普拉斯(De’Moivre-Laplace)中心极限定理，这列二项分布将趋近于正态分布。

   也就是说泊松分布和正态分布都来自于二项分布

3. 实际运用中当 n 很大时一般都用正态分布来近似计算二项分布，但是如果同时 np 又比较小（比起n来说很小），那么用泊松分布近似计算更简单些，毕竟泊松分布跟二项分布一样都是离散型分布。







# 多维正态分布（multivariate normal distribution）

我们先看一维的正态分布：
$$
N\left(x| \mu, \sigma^{2}\right)=\frac{1}{\sqrt{2 \pi} \sigma} \exp \left(-\frac{1}{2 \sigma^{2}}(x-\mu)^{2}\right)
$$
大家都没问题哈！



x 是 d 维的向量， $\Sigma$ 是 x 的协方差矩阵
$$
N(x | u, \Sigma)=\frac{1}{(2 \pi)^{d/2}{|\Sigma|^{1 / 2}}} \exp \left[-\frac{1}{2}(x- u)^{T} \Sigma^{-1}(x-u)\right]
$$


# 迹

相似矩阵的迹都相等



### 矩阵特征值之和等于矩阵的迹

矩阵A的特征方程如下：

$\operatorname{det}(\lambda I-A)=\left|\begin{array}{cccc}{\lambda-a_{11}} & {-a_{12}} & {\dots} & {-a_{1 n}} \\ {-a_{21}} & {\lambda-a_{22}} & {\dots} & {-a_{2 n}} \\ {\dots} & {\dots} & {\dots} & {\dots} \\ {-a_{n 1}} & {-a_{n 2}} & {\dots} & {\lambda-a_{n n}}\end{array}\right|$

行列式展开，如果想要 $\lambda^{n-1}$ 这一项，只有 $\left(\lambda-a_{11}\right)\left(\lambda-a_{22}\right) \ldots\left(\lambda-a_{n n}\right)$ ，

那么可以得到 $\lambda^{n-1}$ 的系数为 $-\left(a_{11}+a_{12}+\ldots+a_{n n}\right)$ ，

上面的特征方程又可以写为特征值的形式 $\operatorname{det}(\lambda I-A)=\left(\lambda-\lambda_{1}\right)\left(\lambda-\lambda_{2}\right) \ldots\left(\lambda-\lambda_{n}\right)$ ，

$\lambda^{n-1}$ 这一项的系数又恰好是 $-\left(\lambda_{1}+\lambda_{2}+\ldots+\lambda_{n}\right)$ 

所以 $\operatorname{tr}(A)=\sum_{k=1}^{n} \lambda_{k}$



# 参数估计

## 最大似然估计

用已知的样本值去估计某一概率分布模型的参数值。
$$
L\left(\theta | x_{1}, x_{2}, \cdots, x_{n}\right)=f\left(x_{1}, x_{2}, \cdots, x_{n} | \theta\right)=\prod f\left(x_{i} | \theta\right)
$$

$$
l(\theta) = \ln L\left(\theta | x_{1}, x_{2}, \cdots, x_{n}\right)=\sum_{i=1}^{n} f\left(x_{i} | \theta\right)
$$

### 推导

#### 伯努利分布（Bernoulli）

每一个样本的概率可以表示为：$p^{x}(1-p)^{1-x}$ ，$x=0 or 1 $  ，$p$是成功的概率

我们现在有n个样本，$D = \{ x_1, x_2,...,x_n \}$ ，把要估计的成功概率 $p$ 记为参数 $\theta$ 

写出对数最大似然估计表达式：
$$
\begin{align}
l(\theta)&=\sum_{i=1}^{n}\log{p(x_i|\theta)} \\
  &= \sum_{i=1}^{n}\log{\theta^{x_i}(1-\theta)^{1-x_i}} \\
  &= \sum_{i=1}^{n}x_i \log{\theta}+(1-x_i)\log(1-\theta) \\
  &= (\sum_{i=1}^{n}x_i)\log{\theta}+(\sum_{i=1}^{n}(1-x_i))\log{(1-\theta)} \\
  &= m\log{\theta} + (n-m)\log(1-\theta)
\end{align}
$$
对 $L(\theta)$ 进行求导，解得 $\theta = \frac{1}{n}\sum_{i=1}^{n}x_i$

#### 正态分布（Gaussian）



#### 例子

> 问：一个不透明的袋中有黑白两种球，数量和比例都不知道。现在我们随机从袋中取球，一共取了100次，有70次是白球。问白球的比例是多少？

> 答：显然这是一个二项分布的模型，那么假设二项分布的参数 袋中白球 的概率为 $\theta$，黑球则为 $(1-\theta)$ 。样本为 $D = \{ x_1, x_2, x_3,...,x_{100} \}$ 
>
> 写出似然函数：$p(D|\theta) = \prod_{k=0}^{100} p(x_k|\theta)={\theta}^{70}(1-\theta)^{30}$
>
> 求出上面似然函数最大时的 $\theta$ 值，(求对数再求导) ，得到 $\theta = 0.7$



#### 计算步骤

1. 选择参数模型（伯努利、高斯），设参数为 $\theta$
2. 获得样本数据  $D = \{ x_1, x_2, x_3,...,x_{n} \}$ 
3. 运用最大似然公式求解



## 贝叶斯估计

在最大似然估计的例子中，如果样本数量不够多，其实存在着很大的问题

> 一个最

贝叶斯估计要解决的不是如何估计参数，而是用来估计新测量数据出现的概率，对于新出现的数据 $x_i$



### 计算步骤

1、计算后验概率 $p(\theta|D)$

根据贝叶斯定理，

 ## 最大后验估计







## 参考

推荐一篇非常强大的文章 [链接](http://www.math.wm.edu/~leemis/2008amstat.pdf)

