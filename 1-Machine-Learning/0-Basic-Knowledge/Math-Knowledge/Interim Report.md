# Interim Report



**what you have achieved in the past weeks**

I have read a lot of related papers about Game AI and Generative Adversarial Networks. I combed through the milestones of some related works. My main work is based on a paper named *Evolving Mario Levels in the Latent Space of a Deep Convolutional Generative Adversarial Network* published on GECCO 2018. This work first introduced Generative Adversarial Networks to solve the problem of video games level procedurally generation. Before this some works including Markov Chains and LSTMs have been implemented. When GANs invented by Lan Goodfellow in 2014, it has been popular in many fields. A simple example is that a GAN trained on photographs can generate new photographs which look at least superficially authentic to human observers, having many realistic characteristics. It’s nature to think that we could apply it on level generation because they have similarity. 

I have been familiar with the AI framework of Super Mario Bro. It provide some codes to simulate a good interface for playing the game with planning algorithms and generating levels. My main work is level procedurally generation but I still need a platform to help assess whether a level is playable and get some feedback(some parameters that will be used for the objective functions). The playing agent I used was the champion Robin Baumgarten’s A* agent which have won the 2009 Mario AI competition. 

I have learned GANs basic algorithm. An interesting description is that the principle can be seen as two-player adversarial game where a generator(faking samples decoded from a random noise vector) and a discriminator(distinguishing whether fake samples or not) are trained at the same time by playing against each other. 

**what will you do in the coming weeks till Week 13**

I will do some experiment to validate my approach. 





Tuning the Multi-Objective on Evolving Mario Levels with Single Generative Adversarial Networks



## Abstract

Generally, Generative Adversarial Networks (GANs) are applied on automatically generating images from training examples. Procedural Content Generation (PCG) of levels for video games could benefit from such machine learning algorithm. Related works had been successfully applied to Super Mario Bro and DOOM.  They also use the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) to help effectively search specific levels that meet some given objectives. However, disadvantage of exist method is that the loss function is single or just weighted sums of some objectives. In this paper we use Multi-Objective Optimization Algorithms to find a trade-off between multiple objectives. On the same time we use a more efficient GAN model named SinGAN to improve the results. Finally we validate our approach using the game *Super Mario Bros*. Experiments present the benefits of such an approach via rating by volunteers subjectively and comparing loss function objectively. 

## Introduction













The framework can easily be extended to incorporate more objectives, as the objectives are given as inputs to the problem. To







One potential area of future work is the use of Multi-Objective Optimization Algorithms [4] to evolve the latent vector using multiple evaluation criteria. Many different criteria can make video game levels enjoyable to different people, and a typical game needs to contain a variety of different level types, so evolving with multiple objective functions could be beneficial. Given such functions, it would also be interesting to compare our results with other procedurally generated content, as well as manually designed levels, in terms of the obtained values. However, further work on automatic game evaluation is required to define purposeful fitness functions.



The rest of this paper is structured as follows. Section 2 introduces the background and related work. The main approach is described in Section 3. Section 4 details the experimental design. The experimental results are presented and discussed in Section 5. Section 6 then concludes the paper.

## BACKGROUND AND RELATED WORK

### 1 Level Generation for the Mario AI Framework



### 2 Generative Adversarial Networks





### 3 Latent variable evolution



### 4 CMA-ES







## Method





















DeepMasterPrints: Generating MasterPrints for Dictionary Attacks via Latent Variable Evolution

–> 

Evolving Mario Levels in the Latent Space of a Deep Convolutional Generative Adversarial Network

–> 

Searching the Latent Space of a Generative Adversarial Network to Generate DOOM Levels





