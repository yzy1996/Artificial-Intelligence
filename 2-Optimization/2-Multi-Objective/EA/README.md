EA 进化算法包括了

genetic algorithms (GA)

evolutionary strategies (ES)

neuro evolution



# 区分 遗传算法GA 和 进化策略ES

- 选好父母进行繁殖 (`GA`); 先繁殖, 选好的孩子 (`ES`)
- 通常用二进制编码 DNA (`GA`); 通常 DNA 就是实数, 比如 1.221 (`ES`)
- 通过随机让 1 变成 0 这样变异 DNA (`GA`); 通过正态分布(Normal distribution)变异 DNA (`ES`)

in GA , x is a sequence of binary codes $x \in \{0, 1\}^n$

while in ES, x is a vector of real numbers, $x \in \mathbb{R}^{n}$



Evolutionary algorithms refer to a division of population-based optimization algorithms inspired by *natural selection*



Our target is to optimize a function $f(x)$, normally is a minimization problem.

- evolutionary algorithm
- gradient descent



## EA

### ES

probility distribution over x, and parameterized by $\theta$, $p_{\theta}(x)$

1. generate a population of samples $D = \{x_i, f(x_i)\}$ where $x_i \sim p_{\theta}(x)$
2. evaluate the fitness of samples in D
3. select the best subset of individuals and use them to update $\theta$



**Simple Gaussian Evolution Strategies**

It models $p_{\theta}(x)$ as a n-dimensional Gaussian distribution
$$
\theta = (\mu, \sigma), p_{\theta}(x) \sim \mathcal{N}\left(\mu, \sigma^{2} I\right)
$$

1. Initialize $\theta$

2. Generate the offspring population of size $\Lambda$ by sampling from Gaussian distribution.

3. Select a top subset of $\lambda$ with optimal $f(x_i)$ and this subset is called elite set.

4. Update new mean and std
   $$
   \begin{aligned}
   \mu^{(t+1)} &=\operatorname{avg}\left(D_{\text {elite }}^{(t+1)}\right)=\frac{1}{\lambda} \sum_{i=1}^{\lambda} x_{i}^{(t+1)} \\
   \sigma^{(t+1)^{2}} &=\operatorname{var}\left(D_{\text {elite }}^{(t+1)}\right)=\frac{1}{\lambda} \sum_{i=1}^{\lambda}\left(x_{i}^{(t+1)}-\mu^{(t+1)}\right)^{2}
   \end{aligned}
   $$



**Covariance Matrix Adaptation Evolution Strategies (CMA-ES)**

The std $\sigma$ accounts for the level of exploration. In vanilla ES, $\sigma^{(t+1)}$ is highly correlated with $\sigma^{(t)}$.  
$$
\theta = (\mu, \sigma, C), p_{\theta}(x) \sim \mathcal{N}\left(\mu, \sigma^{2} C\right)
$$






### GA

$x$ is a sequence of binary codes, $x \in \{0, 1\}^n$

