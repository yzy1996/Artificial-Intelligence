One sentence to summary: the latent space of GANs have semantically meaningful directions.

Which results moving in these directions corresponds to human-interpretable image transformations.

Examples: rotation, zooming or recoloring, 

exploitation of these directions would make image editing more straightforward



Methods:

- supervised (require human labels, pre-trained models)

  {Interpreting the latent space of gans for semantic face editing}

  {Ganalyze: Toward visual definitions of cognitive image properties}

- self-supervised (image augmentations) - [simple transformations]

  {On the”steerability” of generative adversarial networks}

  {Controlling generative models with continuos factors of variations}

- unsupervised ()

  {Unsupervised Discovery of Interpretable Directions in the GAN Latent Space}

>前两者can only discover researchers expectation directions. 需要想象力
>
>后者能实现你所想不到



效果是：

orthogonal image transformations

different directions do not interfere with each other





The key of interpreting the latent space of GANs is to find the meaningful subspaces corresponding to the human-understandable attributes. Through that, moving the latent code
towards the direction of a certain subspace can accordingly change the semantic occurring in the
synthesized image. However, due to the high dimensionality of the latent space as well as the large diversity of image semantics, finding valid directions in the latent space is extremely challenging.



## supervised learning based approaches

randomly sample a large amount of latent codes, then synthesize corresponding images and annotate them with labels, and finally use these labeled samples to learn a separation boundary in the latent space.

存在的问题：需要预定义的语义，需要大量采样



