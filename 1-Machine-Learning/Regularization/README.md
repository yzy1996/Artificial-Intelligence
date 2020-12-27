Recent works have shown deep neural networks (DNN) are greatly overparametrized as they can be pruned significantly without any loss in accuracy.

[Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding]()

[Soft weight-sharing for neural network compression]()

[Variational dropout sparsifies deep neural networks]()

DNN can easily overfit if not properly regularized.

[Understanding deep learning requires rethinking generalization]()





To solve it, we can employ model compression and sparsification techniques. A straightforward approach is the L0 norm regularization. However, this causes an intractable optimization because of non-differentiable.



[Learning sparse neural networks through L0 regularization]() propose a framework for surrogate L0 regularized objectives. 



transforming continuous random variables (r.v.s) with a hard-sigmoid

> hard-sigmoid

$$
f(x) = \text{clip}(\frac{x+1}{2}, 0, 1) = \max \left(0, \min \left(1, \frac{(x+1)}{2}\right)\right) = 
\begin{cases}
  0, & x \le 1 \\
  \frac{x+1}{2}, &  0< x < 1 \\
  1, & x \ge 0
\end{cases}
$$



A normal L0 regularized empirical risk minimization procedure can be shown as:
$$
\min_\boldsymbol{\theta} \mathcal{R}(\boldsymbol{\theta})=\frac{1}{N}\left(\sum_{i=1}^{N} \mathcal{L}\left(h\left(\mathbf{x}_{i} ; \boldsymbol{\theta}\right), \mathbf{y}_{i}\right)\right)+\lambda\|\boldsymbol{\theta}\|_{0}, \quad\|\boldsymbol{\theta}\|_{0}=\sum_{j=1}^{|\theta|} \mathbb{I}\left[\theta_{j} \neq 0\right]
$$
