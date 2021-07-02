# <p align=center>`Meta Learning`</p>

> The order is from the latest to the old


<span id="LEO"></span>
[Meta-Learning with Latent Embedding Optimization](https://arxiv.org/pdf/1807.05960.pdf)  
**[`ICLR 2019`] (DeepMind)**  
*Andrei A. Rusu, Dushyant Rao, Jakub Sygnowski, Oriol Vinyals, Razvan Pascanu, Simon Osindero, Raia Hadsell*

<details><summary>Click to expand</summary><p>

![image-20210629134043053](https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210629134051.png)

> **Summary**

Solve the problem of "high-dimensional parameter spaces in extreme low-data regimes".

So they learn a data-dependent latent generative representation of model parameters.

Latent embedding optimization (LEO) decouples the gradient-based adaptation procedure from the underlying high-dimensional space of model parameters.

> **Details**

**dataset.** miniImageNet, tieredImageNet

Encoding process: 

$g_{\phi_e}(\mathbf{x})$ 
$$
\boldsymbol{\mu}_{n}^{e}, \boldsymbol{\sigma}_{n}^{e}=\frac{1}{N K^{2}} \sum_{k_{n}=1}^{K} \sum_{m=1}^{N} \sum_{k_{m}=1}^{K} g_{\phi_{r}}\left(g_{\phi_{e}}\left(\mathrm{x}_{n}^{k_{n}}\right), g_{\phi_{e}}\left(\mathrm{x}_{m}^{k_{m}}\right)\right)
$$
Decoding process:
$$

$$


</p></details>

---

