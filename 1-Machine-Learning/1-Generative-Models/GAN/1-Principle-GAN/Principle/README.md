# Principle

Since GANs invention, it has yield impressive results, especially for image generation.

Recent work can synthesize random high-resolution portraits that are often indistinguishable from real faces.



Generative models aim to approximate samples from a complex high-dimensional target distribution $\mathbb{P}$. 

The adversarial mechanism reflects by a generator and a discriminator who compete against each other. Unlike other deep neural network models trained with a loss function until convergence, GAN train these two together to maintain a equilibrium finally.

The generator learns to map from a low-dimension space $\mathcal{Z}$ to a high-dimension space $\mathcal{X}$ with a model distribution $\mathbb{Q}$.

The discriminator learns to accurately distinguish between the synthesized data $\mathbf{Y}$ coming from $\mathbb{Q}$ and the real data $\mathbf{X}$ from $\mathbb{P}$. 





## Convergence











## Distance metrics

> This topic studies the distance between target distribution $P_x$ and model distribution $P_z(G)$ 



$z$ is a distribution and $x$ is a distribution and $x^{\prime}$ is also a distribution



 high-dimensional 



integral probability metrics (IPMs)

> a “well behaved” function with large amplitude where $P_x$ and$P_z$ differ most

- Wasserstein IPMs



Maximum Mean Discrepancies (MMDs)

> the critic function is a member of a reproducing kernel Hilbert space



[On gradient regularizers for MMD GANs](https://arxiv.org/pdf/1805.11565.pdf)

**`[NeurIPS 2018]`**	**`(UCL)`**	**`[Michael Arbel, Arthur Gretton]`**	**([:memo:]())**	**[[:octocat:]()]**

<details><summary>Click to expand</summary><p>



</p></details>

---

