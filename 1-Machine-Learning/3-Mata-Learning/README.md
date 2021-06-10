English | [简体中文](./README.zh-CN.md)

<h1 align="center">Meta Learning</h1>
<div align="center">

## Introduction

Meta-learning typically addresses the problem of **few-shot learning**, where some examples of a given task are used to learn an algorithm that achieves better performance on new, previously unseen instances of the same task.

The goal is to train a learner that can quickly adapt to new, unseen tasks given only few training examples, often referred to as context observations.

The goal of meta-learning is to learn from previous tasks a well-generalized meta-learnerM() which can facilitate the training of the base learner in a future task with a few examples.

> An example in computer vision: a network need to learn to differentiate between new classes based on only a small number of labeled instances of each class.



take advantages of the knowledge learned from related tasks

few-shot



- learn quickly with a few samples
- 



What is meta loss function:

1. cumulative regret: $L(\phi) = \sum_{t=1}^T \sum_{j=1}^k f(x_j^t)$.

   drawbacks: does not reflect any exploration

   

Branch

- learn an optimizer or update rule

- conditional and attentive neural processes

- learn the initialization of a neural network

  > can specialize to a new task via few steps of gradient descent



Meta-Learning frame:

N-way and K-shot few-shot task "Problem definition"

given a labeled dataset of base classes $C_{base}$ with a large amount of images in each class, the goal is to learn concepts in novel classes $C_{novel}$ with a few samples in each class. 

(training) We have N classes, in each class, there are K support samples and Q query samples

the goal is to classify the query samples after training from support samples.

(evaluation) compute from many tasks samples from the data in $C_{novel}$

meta-learning architectures can be categorized into:

- Memory-based methods.
- Optimization-based methods.
- Metric-based methods.



### MAML

$$
\theta_0^{j+1} = \theta_0^j - \beta \nabla_{\theta}L(\theta_m(\theta, T_j))|_{\theta=\theta_0^j}
$$

*outer loop* vs *inner loop*

### Reptile

$$
\theta_0^{j+1} = \theta_0^j - \beta (\theta_m(\theta_0^{j}, T_j) - \theta_0^j)
$$





## Literature

- review

2020-Meta-learning in neural networks: A survey





### MAML



[Hierarchically Structured Meta-learning](https://arxiv.org/pdf/1905.05301.pdf)



> **Summary**

