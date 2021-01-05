# Representation Learning



## Motivation



因为它是如此地重要

所以将它专门发展成一个小领域 

Representation learning has become a field in itself in the machine learning community



*Representation learning*, i.e., learning representations of the data that make it easier to extract useful information when building classifiers or other predictors [^1].



In the case of probabilistic models, a good representation is often one that captures the posterior distribution of the underlying explanatory factors for the observed input.



**What is representation learning?**

> The performance of machine learning methods heavily depend on the choice of data representation (or features). For this reason, much effort involve preprocessing pipelines (regularization) and data transformations (kernel) that result in a representation of the data that enhance ML algorithms.
>
> regression and classification both can be seen as a process of finding representation of data.







traditional feature extraction need human ingenuity and prior knowledge. It would be highly desirable to make ML algorithms less dependent on this manual process. We hope it could automatically extract the discriminative information from the data.







机器学习，学习的是什么，是提取特征并根据特征分类或者回归

基本上都是为了



**提取特征**

extract the discriminative information from the data

learn to identify and disentangle the underlying explanatory factors hidden in the observed milieu of data.



**解耦**

Complex data arise from the rich interaction of many sources. These factors interact in a complex web that can complicate AI-related tasks such as object classification



**我们希望表征是什么样的呢，达到什么效果呢？**

representations to 

- disentangle the factors of variation



the approach we adopt for overcoming these challenges must leverage the data itself, using vast quantities of unlabeled examples, to learn representations that separate the various explanatory sources.



the most robust approach to feature learning is to disentangle as many factors as possible, discarding as little information about the data as is practical.



**有哪些表征学习的手段**



### Application

2D

3D



[^1]: Representation Learning: A Review and New





Bengio et al. [3] propose that a disentangled representation is one for which changes in the encoded data are sparse over real-world transformations; that is, changes in only a few latents at a time should be able to represent sequences which are likely to happen in the real world.