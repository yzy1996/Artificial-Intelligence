# Loss function of GANs

原始的GAN是

non-saturating GAN loss



saturating GAN loss





| Name (paper link) |                        Loss Function                         |
| :---------------: | :----------------------------------------------------------: |
|  **Vanilla GAN**  |                    $$L_D^{GAN} = E\\as$$                     |
|     **LSGAN**     |                                                              |
|     **WGAN**      | $$\begin{align}&L_{D}^{WGAN}=-\mathbb{E}_{x \sim p_{\text {data }}}[D(x)]+\mathbb{E}_{z \sim p(z)}[D(G(z))]\\&L_{G}^{WGAN}=-\mathbb{E}_{z \sim p(z)}[D(G(z))]\\&W_D \leftarrow \text{clip_by_value}(W_D, -0.01, 0.01) \end{align}$$ |
|    **WGAN-GP**    | $$L_{D}^{WGAN-GP}=L_{D}^{WGAN} + \lambda \mathbb{E}[(|\nabla_xD(x)|-1)^2 ]\\L_{G}^{WGAN-GP}=L_{G}^{WGAN}$$ |
|                   |                                                              |

> inspired by [hwalsuklee](https://github.com/hwalsuklee/tensorflow-generative-model-collections)


