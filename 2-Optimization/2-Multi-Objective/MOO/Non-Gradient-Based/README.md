# 1. surrogate gradient

代理梯度

true gradient is inaccessible 

- unknown or not defined: discrete stochastic variables

- hard or expensive to compute: truncated backprop through time





surrogate gradient means directions directions that may be correlated with, but not necessarily identical to, the true gradient. 







### Vanilla ES

minimize a function $f(x)$ over a parameter space in $n$-dimensions $x \in \mathbb{R}^{n}$