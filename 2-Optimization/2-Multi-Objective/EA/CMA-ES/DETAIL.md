[toc]

## 基础知识

**采样的基本公式**
$$
x_i^{t+1} \sim \mathcal{N}\left(\mu^t, (\sigma^t)^2 C^t \right) = \mu^t + \sigma^t \mathcal{N}\left(0, C^t\right)
$$
其中 $x \in \mathbb{R}^n$ , $\mu \in \mathbb{R}^n$  , $\sigma \in \mathbb{R}$ , $C \in \mathbb{R}^{n \times n}$



## 初始化状态

**我们拥有什么**

对于一个初始化的分布 $\mathcal{N}(\mu^0, \sigma^0 C^0)$ ，采样 $\Lambda$ 个样本点

$$
x_1^{t+1}, x_2^{t+1}, \dots, x_{\Lambda}^{t+1}
$$

经过目标函数的计算并排序：
$$
f(x_{1:\Lambda}^{t+1}) \le f(x_{2:\Lambda}^{t+1}) \le \cdots \le f(x_{\Lambda:\Lambda}^{t+1})
$$
选择前 $\lambda$ 个优秀的样本点：

$$
x_{1:\Lambda}^{t+1}, x_{2:\Lambda}^{t+1}, \dots, x_{\lambda:\Lambda}^{t+1}
$$



## 开始更新

### 更新均值

新的搜索分布的均值是 $\lambda$ 个选择点的加权和：
$$
\begin{gather}
\mu^{t+1} = \sum_{i=1}^{\lambda} w_i x_{i:\lambda}^{t+1}
\\
\sum_{i=1}^{\lambda} w_i = 1, w_1 \ge w_2 \ge \dots \ge w_{\lambda} \ge 0
\end{gather}
$$


我们将公式(1)写成一个广义的公式：
$$
\mu^{t+1} = \mu^t + \alpha_{\mu} \sum_{i=1}^{\lambda} w_i (x_{i:\lambda}^{t+1} - \mu^t)
$$
当 $\alpha_{\mu} = 1$ 时，公式(2) 和 (4) 时一样的；而当 $\alpha_{\mu} \le 1$ 时，能有效对抗带噪声的目标函数。



### 更新协方差矩阵

通过采样得到的点，我们可以重新估计原来的协方差矩阵。这样得到的是一个经验上的协方差矩阵(empirical covariance matrix)，需要无偏修正：
$$
C_{emp}^{t} = \frac{1}{\Lambda - 1} \sum_{i=1}^{\Lambda} \left(\frac{x_i^{t+1} - \frac{1}{\Lambda} \sum_{j=1}^\Lambda x_j^{t+1}}{\sigma^t} \right) \left(\frac{x_i^{t+1} - \frac{1}{\Lambda} \sum_{j=1}^\Lambda x_j^{t+1}}{\sigma^t} \right)^T
$$
如果我们使用上一代真实的样本均值，不需要无偏修正：
$$
\begin{equation}
\begin{aligned}
C_{\Lambda}^{t} 
&= \frac{1}{\Lambda} \sum_{i=1}^{\Lambda} \left(\frac{x_i^{t+1} - \mu^t}{\sigma^t} \right)\left(\frac{x_i^{t+1} - \mu^t}{\sigma^t}  \right)^T\\
&= \frac{1}{\Lambda} \sum_{i=1}^{\Lambda} (y_i^{t+1})(y_i^{t+1})^T
\end{aligned}
\end{equation}
$$
(5)可以理解成估计采样点的方差，(6)是估计采样步骤的方差

现在我们想估计被挑选的优解的新协方差矩阵：
$$
C_{\lambda}^{t+1} = \sum_{i=1}^{\lambda} w_i \left(\frac{x_{i:\Lambda}^{t+1} - \mu^t}{\sigma^t} \right)\left(\frac{x_{i:\Lambda}^{t+1} - \mu^t}{\sigma^t} \right)^T
$$
使用 Estimation of Multivariate Normal Algorithm 的估计：
$$
C_{EMNA_{global}}^{t+1} = \frac{1}{\lambda} \sum_{i=1}^{\lambda} \left(\frac{x_{i:\Lambda}^{t+1} - \frac{1}{\lambda} \sum_{j=1}^\lambda x_{j:\Lambda}^{t+1}}{\sigma^t} \right) \left(\frac{x_{i:\Lambda}^{t+1} - \frac{1}{\lambda} \sum_{j=1}^\lambda x_{j:\Lambda}^{t+1}}{\sigma^t} \right)^T
$$


**Rank-$\lambda$-Update**

为了实现更加快速的搜索，种群数量 $\lambda$ 必须很小，因此上述估计方法就将不准确。一个更有效的方式是累积使用历史信息：
$$
C^{t+1} = \frac{1}{t + 1} \sum_{i=0}^t \frac{1}{(\sigma^i)^2}C_{\lambda}^{i + 1}
$$
上式每一步都使用了相同的权重，但更有效的方式是给临近的更新一个更大的权重，因此每步迭代写成：
$$
\begin{equation}
\begin{aligned}
C^{t+1} 
&= (1 - \alpha_{\lambda})C^t + \alpha_{\lambda} \frac{1}{(\sigma^t)^2} C_{\lambda}^{t+1}\\
&= (1 - \alpha_{\lambda})C^t + \alpha_{\lambda} \sum_{i=1}^{\lambda} w_i (y_{i:\Lambda}^{t+1}) (y_{i:\Lambda}^{t+1})^T
\end{aligned}
\end{equation}
$$
上式又可以推广到 $\Lambda$ 个权重的情况
$$
C^{t+1} = (1 - \alpha_{\lambda} \sum_{i=1}^{\Lambda} w_i)C^t + \alpha_{\lambda} \sum_{i=1}^{\Lambda} w_i (y_{i:\Lambda}^{t+1}) (y_{i:\Lambda}^{t+1})^T
$$


**Rank-1-Update**

前面我们用 $\lambda$ 个点更新，现在我们只用1个点更新
$$
\begin{equation}
\begin{aligned}
C^{t+1} 
&= (1 - \alpha_1) C^t + \alpha_1 \left(\frac{x_{1:\Lambda}^{t+1} - \mu^t}{\sigma^t} \right) \left(\frac{x_{1:\Lambda}^{t+1} - \mu^t}{\sigma^t} \right)^T\\
&= (1 - \alpha_1) C^t + \alpha_1 (y_{i:\Lambda}^{t+1})(y_{i:\Lambda}^{t+1})^T
\end{aligned}
\end{equation}
$$


因为前面 (10) (11) 都用到了 $yy^T$ ，又因为 $yy^t = (-y)(-y)^T$ 这一形式会忽略符号信息，所以希望采取一种办法能弥补这一信息遗漏。

这里引入一个新的概念-进化路径(evolution path)，记作 $p_c$ ，它被表示为连续步骤的和，例如一个3步的过程：
$$
\frac{\mu^{t+1} - \mu^t}{\sigma^t} + \frac{\mu^{t} - \mu^{t-1}}{\sigma^{t-1}} + \frac{\mu^{t-1} - \mu^{t-2}}{\sigma^{t-2}}
$$
进化路径的更新公式为：
$$
p_c^{t+1} = (1 - \alpha_c) p_c^t + \sqrt{\alpha_c(2 - \alpha_c)}\frac{\mu^{t+1} - \mu^t}{\sigma^t}
$$

> 为什么使用 $\sqrt{\alpha_c (2 - \alpha_c)}$ 作为因子呢？
>
> 是因为如果 
>
> $$
> p_c^t \sim \frac{x_{i:\Lambda}^{t+1} - \mu^t}{\sigma^t} \sim \mathcal{N}(0, C)  \ \ \text{for all}\  i=1, \dots, \lambda
> $$
> 又 $(1 - \alpha_c)^2 + {\sqrt{\alpha_c(2 - \alpha_c)}}^2 = 1$，所以
> $$
> p_c^{t+1} \sim \mathcal{N}(0, C)
> $$
> 即 $p_c$ 是共轭的



所以通过进化路径更新C的公式为：
$$
C^{t+1} = (1 - \alpha_1) C^t + \alpha_1 (p_c^{t+1})(p_c^{t+1})^T
$$


当 $\alpha_c = 1, \lambda = 1$ 时，(17)(12)(10) 是相同的



**综合 rank-$\lambda$-update 和 Rank-1-Update**
$$
C^{t+1} = (1 - \alpha_1 - \alpha_{\lambda}) C^t + \alpha_1 (p_c^{t+1})(p_c^{t+1})^T + \alpha_{\lambda} \sum_{i=1}^{\Lambda} w_i (y_{i:\Lambda}^{t+1}) (y_{i:\Lambda}^{t+1})^T
$$

### 更新步长

我们同样使用一条进化路径来估计步长，记作 $p_{\sigma}$


$$
\sigma^{t+1} = \sigma^{t} \exp\left(\frac{\alpha_{\sigma}}{d_{\sigma}}\left(\frac{\left\|p_{\sigma}^{(t+1)}\right\|}{\mathbb{E}\|\mathcal{N}(0, I)\|}-1\right)\right)
$$





## 整体算法流程

