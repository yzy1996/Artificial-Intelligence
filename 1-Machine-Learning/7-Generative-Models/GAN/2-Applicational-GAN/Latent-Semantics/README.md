interpretable control directions



## 存在性

2015 Radford et al. find GAN latent space processes semantically meaningful vector space arithmetic





## 意义：

1. browse through the concepts that the GAN has learned
2. training a general model requires enormous computational resources, so interpret and extend the capabilities of existing GANs

对象：existing GANs







**One sentence to summary**: the latent space of GANs have semantically meaningful directions.

Which results moving in these directions corresponds to human-interpretable image transformations.

**Examples**: rotation, zooming or recoloring, 

exploitation of these directions would make image editing more straightforward



### Semantic image editing

> task is to transform a source image to a target image while modifying desired semantic attributes.

> for artistic visualization, design, photo enhancement

two primary goals

> providing continuous manipulation of multiple attributes simultaneously
>
> maintaining the original image’s identity as much as possible while ensuring photo-realism



Existing GAN-based approaches can be categorized roughly into two groups:

1) image-space editing

> These approaches often have high computational cost, and they primarily focus on binary attribute (on/off) changes, rather than providing continuous attribute editing abilities

2) latent-space editing

> lower-dimensional space
>
> 







## 主要方法

- [Supervised]() (require human labels, pre-trained models)

  {Interpreting the latent space of gans for semantic face editing}

  {Ganalyze: Toward visual definitions of cognitive image properties}

- [Self-supervised]() (image augmentations) - [simple transformations]

  {On the”steerability” of generative adversarial networks}

  {Controlling generative models with continuos factors of variations}

- [Unsupervised]() ()

  {Unsupervised Discovery of Interpretable Directions in the GAN Latent Space}
  
  {Ganspace: Discovering interpretable gan controls}





>前两者can only discover researchers expectation directions. 需要想象力
>
>后者能实现你所想不到



效果是：

orthogonal image transformations

different directions do not interfere with each other





The key of interpreting the latent space of GANs is to find the meaningful subspaces corresponding to the human-understandable attributes. Through that, moving the latent code towards the direction of a certain subspace can accordingly change the semantic occurring in the synthesized image. However, due to the high dimensionality of the latent space as well as the large diversity of image semantics, finding valid directions in the latent space is extremely challenging.



### Supervised Learning

> domain-specific transformations (adding smile or glasses)

randomly sample a large amount of latent codes, then synthesize corresponding images and annotate them with labels, and finally use these labeled samples to learn a separation boundary in the latent space.

存在的问题：需要预定义的语义，需要大量采样



> Shen et al. Interpreting the latent space of gans for semantic face editing

> Karras et al. A style-based generator architecture for generative adversarial networks

Use the classifiers pretrained on the CelebA dataset to predict certain face attributes

Add labels to latent space and separate a hyperplane. A normal to this hyperplane becomes a direction that captures the corresponding attribute.



> Controlling generative models with continues factors of variations

solve the optimization problem in the latent space that maximizes the score of the pretrained model, predicting image memorability



**weakness**: need human labels or pretrained models, expensive to obtain



### Self-supervised Learning

> domain agnostic transformations (zooming or translation)



> Jahanian et al. On the”steerability” of generative adversarial networks
>
> Plumerault et al. Controlling generative models with continuos factors of variations

simple image augmentations such as zooming or translation 



