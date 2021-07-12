# <p align=center>`Hard Constraints` </p>

To learn a more robust and generalizable model.

## Definition

Consider a set of training samples given as $\{x^{(i)}, y^{(i)}\}_{i=1}^m$. Given the parameters $w$ of the network, let $L(w)=\frac{1}{m} \sum_{i=1}^m (y^{(i)} - \hat{y}^{(i)})^2$ denote the loss function. The goal of learning a neural network is to find a set of parameters satisfied $w^* = \arg\min_w L(w)$.

Constrained Learning of Neural Models need to impose a set of hard constraints which hold over the output label space. These background knowledges will help to learn a more robust and generalizable model.

Given a set of $K$  constraints as $\{C_1(w), C_2(w), \cdots, C_K(w)\}$, where $C_k(w): \{f_k(w \le 0\}$. Each constraint is a function of the predicted value $\hat{y}$ on a given example $x$.

A Lagrangian-based Formulation:

$$
\arg\min_{w} L(w) \text { subject to } \quad f_{k}^{i}(w) \leq 0 ; \quad \forall 1 \leq i \leq m ; \quad \forall 1 \leq k \leq K
$$


## Methodology

- modeled the constraints as soft, incorporating penalty in the loss term
- directly optimize the hard constraints using a Langrangian based formulation







## Literature

[A Primal-Dual Formulation for Deep Learning with Constraints](https://papers.nips.cc/paper/2019/file/cf708fc1decf0337aded484f8f4519ae-Paper.pdf)  
**[`NeurIPS`] (`Indian Institute of Technology Delhi`)** [[Code](https://github.com/dair-iitd/dl-with-constraints)]  
*Yatin Nandwani, Abhishek Pathak, Mausam and Parag Singla*

