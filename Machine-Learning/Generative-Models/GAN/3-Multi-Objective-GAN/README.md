# Multi-Objective GAN

We can construct a framework of Multi-Objective GAN with:
$$
\mathcal{L}_{G}=\sum_{k=1}^{K} \alpha_{k} \mathcal{L}_{k}
$$
where $\mathcal{L}_{k}$ is the loss of objective $k$



Previous works were done around multiple discriminators including 



[Durugkar, 2016, Generative multi-adversarial networks](Generative multi-adversarial networks.pdf)

> The goal of using the proposed methods is to favor worse discriminators, thus providing more useful gradients to the generator during training.
>
> Using a softmax weighted average of K discriminators, 
> $$
> \mathcal{L}_{D_{k}}=-\mathbb{E}_{\mathbf{x} \sim p_{\text {data }}} \log D_{k}(\mathbf{x})-\mathbb{E}_{\mathbf{z} \sim p_{z}} \log \left(1-D_{k}(G(\mathbf{z}))\right)
> $$
> $$
> \mathcal{L}_{G}=\sum_{k=1}^{K} \alpha_{k} \mathcal{L}_{D_{k}}
> $$
> $$
> \alpha_k = \text{softmax}(l_{1:K})_k = \frac{e^{\beta l_k}}{\sum_{i=1}^K e^{\beta l_i}}
> $$
> $\beta$ is a hyperparameter, ($\beta$=0, 1 , $\infty$)
>

[Neyshabur, 2017, Stabilizing GAN Training with Multiple Random Projections](Stabilizing GAN Training with Multiple Random Projections.pdf)

> using average loss minimization
> 
> $$
> \mathcal{L}_{D_{k}}=-\mathbb{E}_{\mathbf{x} \sim p_{\text {data }}} \log D_{k}(\mathbf{x})-\mathbb{E}_{\mathbf{z} \sim p_{z}} \log \left(1-D_{k}(G(\mathbf{z}))\right)
> $$
> $$
> \mathcal{L}_{G}=-\sum_{k=1}^{K} \mathbb{E}_{\mathbf{z} \sim p_{z}} \log D_{k}(G(\mathbf{z}))
> $$
>
> $$
> \alpha_k = \frac{1}{K}
> $$

