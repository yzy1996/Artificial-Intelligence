# Latent Space Image Manipulation



## Introduction

To tackle the instability of the training procedure...



These methods can be divided into two categories:

- ...



## Literature

unsorted

[Challenging common assumptions in the unsupervised learning of disentangled representations](https://arxiv.org/abs/1811.12359)







[GLO](#GLO)









---

<span id="GLO"></span>
[Optimizing the Latent Space of Generative Networks](https://arxiv.org/pdf/1707.05776.pdf)  
**[`ICML 2018`] (`Facebook`)**  
*Piotr Bojanowski, Armand Joulin, David Lopez-Paz, Arthur Szlam*

<details><summary>Click to expand</summary>

> **Summary**

> **Details**

compare the $\ell_2$ loss and the Laplacian pyramid Lap_1 loss, finally use a weighted combination of them.
$$
\operatorname{Lap}_{1}\left(x, x^{\prime}\right)=\sum_{j} 2^{2 j}\left|L^{j}(x)-L^{j}\left(x^{\prime}\right)\right|_{1}
$$
where $L^j(x)$ is the $j$-th level of the Laplacian pyramid representation of $x$. -[ref](Diffusion distance for histogram comparison)



</details>

---

