# ELBO

> Evidence Lower Bound



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

