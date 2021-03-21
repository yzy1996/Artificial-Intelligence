English | [简体中文](./README.zh-CN.md)

<h1 align="center">Meta Learning</h1>
<div align="center">

## Introduction

Meta-learning typically addresses the problem of **few-shot learning**, where some examples of a given task are used to learn an algorithm that achieves better performance on new, previously unseen instances of the same task.

The goal is to train a learner that can quickly adapt to new, unseen tasks given only few training examples, often referred to as context observations.

> An example in computer vision: a network need to learn to differentiate between new classes based on only a small number of labeled instances of each class.



take advantages of the knowledge learned from related tasks

few-shot





What is meta loss function:

1. cumulative regret: $L(\phi) = \sum_{t=1}^T \sum_{j=1}^k f(x_j^t)$.

   drawbacks: does not reflect any exploration

   

Branch

- learn an optimizer or update rule

- conditional and attentive neural processes

- learn the initialization of a neural network

  > can specialize to a new task via few steps of gradient descent





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



