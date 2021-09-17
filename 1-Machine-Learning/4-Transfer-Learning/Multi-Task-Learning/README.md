<h1 align="center">Multi-Task Learning</h1>
<div align="center">
Introduction  
</div>

## Introduction

By sharing representations between related tasks, we can enable our model to generalize better on our original task. This approach is called Multi-Task Learning (MTL).



又名 joint learning, learning to learn, learning with auxiliary task



leverage information learned by one task to benefit the training of other tasks.



### Problem Definition

For a set of $n$ tasks $\mathcal{T} = \left\{\tau_{1}, \tau_{2}, \ldots, \tau_{n}\right\}$



## Literature

[Pareto Multi-Task Learning](https://papers.nips.cc/paper/9374-pareto-multi-task-learning.pdf)

multiple loss:

[Deep Network Interpolation for Continuous Imagery Effect Transition](https://arxiv.org/abs/1811.10515)

[Dynamic-Net: Tuning the Objective Without Re-training for Synthesis Tasks](http://export.arxiv.org/abs/1811.08760)

hypernetworks:

[HyperNetworks](https://arxiv.org/abs/1609.09106)

conditional training to multiple loss:

[You Only Train Once: Loss-Conditional Training of Deep Networks](https://openreview.net/forum?id=HyxY6JHKwr)

hypernetworks to continual learning:

[Continual learning with hypernetworks](https://openreview.net/forum?id=SJgwNerKvB&noteId=rkludeIKiS)



[Efficiently Identifying Task Groupings for Multi-Task Learning](https://arxiv.org/pdf/2109.04617.pdf)  
**[`Arxiv 2021`] (`Google`)**  
*Christopher Fifty, Ehsan Amid, Zhe Zhao, Tianhe Yu, Rohan Anil, Chelsea Finn*



## Code

https://github.com/Xi-L/ParetoMTL



