English | [简体中文](./README.zh-CN.md)

<h1 align="center">Meta Learning</h1>
<div align="center">
## Introduction

Meta-learning typically adresses the problem of few-shot learning, where some examples of a given task are used to learn an algorithm that achieves better performance on new, prevoously unseen instances of the same task.

> An example in computer vision: a network need to learn to differentiate between new classes based on only a small number of labeled instances of each class.



take advantages of the knowledge learned from related tasks

few-shot







What is meta loss function:

1. cumulative regret: $L(\phi) = \sum_{t=1}^T \sum_{j=1}^k f(x_j^t)$.

   drawbacks: does not reflect any exploration

   



Model-Agnostic Meta Learning (MAML)

Reptile