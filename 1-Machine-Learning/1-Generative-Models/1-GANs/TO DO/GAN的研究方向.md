# Research Direction of GAN



GAN minimax optimization still poses great theoretical and empirical challenges.





## model interpretability



### Interpretable directions in the latent space



### Disentanglement learning

[InfoGAN]()  enforces the generated images to preserve information about the latent code coordinates by maximizing the corresponding mutual information.

[$\beta$-VAE]()  put more emphasis on the $KL$-term in the standard VAE's ELBO objective.



[2019-Oogan](Disentangling gan with one-hot sampling and orthogonal regularization)  forces the code vector $c$ to be one-hot, simplifying the task for a GAN discriminators' head to predict the code.

[2020-VAE-GAN](High-fidelity synthesis with disentangled representation)  combine VAE and GAN to achieve a disentanglement images representation by the VAE and then pass the discovered code to the GAN model.













## GANs interpretability

- the structure of latent spaces

  semantic meaningful directions

  1.existence: exist such directions

  2.domain agnostic transformations (zooming or translation) & domain-specific transformations (adding smile or glasses)

- disentangled latent spaces



> On the”steerability” of generative adversarial networks
>
> Controlling generative models with continuos factors of variations
>
> Ganalyze: Toward visual definitions of cognitive image properties
>
> Interpreting the latent space of gans for semantic face editing









## Inverting the synthesis networks

It is easy and better to manipulate a given image in the latent feature space.



for a given image, we need to find a matching latent code first.



related work:

- Image2StyleGAN: How to embed images into the StyleGAN latent space?
- Style generator inversion for image enhancement and animation

> these two suggest that instead of finding a common latent code w, the results improve if a separate w is chosen for each layer of the generator.



applications

