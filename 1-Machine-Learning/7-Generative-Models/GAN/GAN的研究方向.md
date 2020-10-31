# GAN的研究方向





## GANs interpretability

- the structure of latent spaces

  1.existence:

  2.domainagnostic transformations (zooming or translation) & domain-specific transformations (adding smile or glasses)

- disentangled latent spaces



## Inverting the synthesis networks

It is easy and better to manipulate a given image in the latent feature space.



for a given image, we need to find a matching latent code first.



related work:

- Image2StyleGAN: How to embed images into the StyleGAN latent space?
- Style generator inversion for image enhancement and animation

> these two suggest that instead of finding a common latent code w, the results improve if a separate w is chosen for each layer of the generator.