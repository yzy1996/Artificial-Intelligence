# 2D Representation

There models try to model the real world by generating realistic samples from latent representations.



likelihood-based frameworks for deep generative learning:

- normalizing flows
- autoregressive models
- Variational autoencoder (VAE)
  - :yum: fast | tractable sampling | easy-to-access encoding networks 
- deep Energy-based models



adversarial game:

generative adversarial networks (GAN)



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

