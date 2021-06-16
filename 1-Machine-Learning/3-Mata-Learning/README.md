English | [中文](./README.zh-CN.md)

<h1 align="center">Meta Learning</h1>
<div align="center">

## Introduction

Meta-learning typically addresses the problem of **few-shot learning**, where some examples of a given task are used to learn an algorithm that achieves better performance on new, previously unseen instances of the same task.

The goal is to train a learner that can quickly adapt to new, unseen tasks given only few training examples, often referred to as context observations.

The goal of meta-learning is to learn from previous tasks a well-generalized meta-learner which can facilitate the training of the base learner in a future task with a few examples.

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



### Meta-Learning frame:

- We have a task distribution $P(\mathcal{T})$, and then we sample a task $\mathcal{T}_i$ with a dataset $\mathcal{D}_i$.

- From this dataset, we sample a support set $\mathcal{D}_i^s=(\mathbf{X}_i^s, \mathbf{Y}_i^s)=\{(\mathbf{x}_{i,k}^s, \mathbf{y}_{i,k}^s)\}_{k=1}^{N_s}$ and a query set $\mathcal{D}_i^q=(\mathbf{X}_i^q, \mathbf{Y}_i^q)=\{(\mathbf{x}_{i,k}^q, \mathbf{y}_{i,k}^q)\}_{k=1}^{N_q}$.
- during meta-training stage, we train the model $f$ on the meta-training tasks.
- during meta-test stage, the well-trained model $f$ is applied to the new task $\mathcal{T}_t$ with its support set $\mathcal{D_t^s}$ and evaluate the performance on the query set $\mathcal{D_t^q}$.





N-way and K-shot few-shot task "Problem definition"

given a labeled dataset of base classes $C_{base}$ with a large amount of images in each class, the goal is to learn concepts in novel classes $C_{novel}$ with a few samples in each class. 

(training) We have N classes, in each class, there are K support samples and Q query samples

the goal is to classify the query samples after training from support samples.

(evaluation) compute from many tasks samples from the data in $C_{novel}$

meta-learning architectures can be categorized into:

- Memory-based methods.
- Optimization-based methods.
- Metric-based methods.



### Main algorithms

#### gradient-based

model-agnostic meta learning (MAML)

The goal of MAML is to learn initial parameters $\theta^*$ such that one or a few gradient steps on $D^s$ 

it's a bi-level optimization process
$$
\theta^{*} \leftarrow \arg \min _{\theta} \mathbb{E}_{\mathcal{T} \sim p(\mathcal{T})}\left[\mathcal{L}\left(f_{\phi}^{M A M L} ; \mathcal{D}^{q}\right)\right], \text { where } \phi=\theta-\eta \nabla_{\theta} \mathcal{L}\left(f_{\theta}^{M A M L} ; \mathcal{D}^{s}\right)
$$


#### metric-based









### problem：

rely on having a large number of diverse meta-learning tasks



==regularization methods==

explicit regularization

augment tasks by individual training tasks through noise、mixup





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

