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





表征（representation）和重构（reconstruction）一直是不分家的两个研究话题。

核心目标是重构，但就像我看到一幅画面，想要转述给另一个人，让他也想象出这个画面的场景，人会将这幅画抽象为一些特征，例如这幅画是自然风光，有很多树，颜色很绿，等等。然后另一个人再根据这些描述，通过自己预先知道的人生阅历，就能还原这幅画/

或者就像公安在找犯人的时候，需要通过描述嫌疑人画像。是通过一些特征在刻画的。

机器同样也需要这样一套范式，只不过可能并不像人一样的语意理解

为了可解释性，以及可控性，我们是希望机器能按照人能理解的一套特征来

![image-20220612154943172](https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/image-20220612154943172.png)



AutoDecoder





这里又需要提及一下重建loss







参考：

https://www.jeremyjordan.me/variational-autoencoders/

https://www.jeremyjordan.me/autoencoders/
