# 正态分布的指数期望 $E[e^x]$



已知 $X \sim \mathcal{N}(\mu, \sigma^2)$，有$f(x) = \frac{1}{\sqrt{2\pi} \sigma} e ^{-\frac{(x-\mu)^2}{2\sigma^2}}$

根据 $E[g(X)] = \int_{-\infty}^{\infty} g(x)f(x) dx$

因此
$$
\begin{aligned}
E[e^X] 
&= \int_{-\infty}^{\infty} e^x f(x) dx \\
&= \int_{-\infty}^{\infty} e^{x} \frac{1}{\sqrt{2\pi} \sigma} e^{-\frac{(x-\mu)^2}{2\sigma^2}} d x \\
&= \int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi} \sigma} e^{x-\frac{(x-\mu)^2}{2\sigma^2}} d x \\
&= \int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi} \sigma} e^{\frac{2 \sigma^2 x - (x-\mu)^2}{2\sigma^2}} d x \\
&= \int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi} \sigma} e^{\frac{-\left[\sigma^2 - (x-\mu)\right]^2 + \sigma^4 + 2\sigma^2 \mu}{2\sigma^2}} d x \\
&= e^{\mu + \frac{\sigma^2}{2}} \int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi} \sigma} e^{\frac{-\left[(x-\mu) - \sigma^2\right]^2}{2\sigma^2}} d x
\end{aligned}
$$
积分项 令 $z = x - \mu$ 可以写成 $\int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi} \sigma} e^{\frac{-\left(z - \sigma^2\right)^2}{2\sigma^2}} d z$ ，这可以看成是服从 $\mathcal{N}(\sigma^2, \sigma^2)$ 的随机变量的概率密度的全域积分，是个定值=1

所以
$$
E[e^X] = e^{\mu + \frac{\sigma^2}{2}}
$$
