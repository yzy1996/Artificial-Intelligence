# 2D Representation

There models try to model the real world by generating realistic samples from latent representations.



<Generating images with sparse representations> divide deep generative models broadly into three categories:

- Generative Adversarial Networks

  > use discriminator networks that are trained to distinguish samples from generator networks and real examples

- Likelihood-based Model

  > directly optimize the model log-likelihood or the evidence lower bound.

  - Variational autoencoder (VAE) 

    > :yum: fast | tractable sampling | easy-to-access encoding networks 

  - normalizing flows

  - autoregressive models

- Energy-based Models

  > estimate a scalar energy for each example that corresponds to an unnormalized log-probability





Details of GANs please see the [file]()



## VAE

The majority of the research efforts on improving VAEs is dedicated to the statistical challenges, such as:

- reducing the gap between approximate and true posterior distribution
- formulatig tighter bounds
- reducing the gradient noise
- extending VAEs to discrete variables
- tackling posterior collapse
- designing special network architectures
  - previous work just borrows the architectures from the classification tasks



VAEs maximize the mutual information between the input and latent variables, requiring the networks to retain the information content of the input data as much as possible.

Information maximization in noisy channels: A variational approach  
**[`NeurIPS 2017`]**

Deep variational information bottleneck  
**[`ICLR 2017`]**

