To tackle the instability of the training procedure

> Why it's a problem? GAN need to find a Nash equilibrium of a non-convex game in a continuous and high dimensional parameter space.



These methods can be divided into two categories:

- **Normalization**
  - **Spectral normalization** (weight matrices in the discriminator are divided by an approximation of their largest singular value)
- **Regularization**
  - **Wasserstein** (penalize the gradient norm of straight lines between real data and generated data)
  - (directly regularize the squared gradient norm for both the training data and the generated data.)





[CR-GAN](#CR-GAN)



### CR-GAN

[Consistency regularization for generative adversarial networks](https://arxiv.org/pdf/1910.12027.pdf)

**`[ICLR 2020]`**	**`(Google)`**	**`[Han Zhang, Honglak Lee]`**	**([:memo:]())**	**[[:octocat:]()]**

<details><summary>Click to expand</summary><p>


**Summary**

> They propose a training stabilizer based on **consistency regularization**. In particular, they **augment data** passing into the GAN discriminator and **penalize the sensitivity** of the discriminator to these augmentations.


</p></details>

---

