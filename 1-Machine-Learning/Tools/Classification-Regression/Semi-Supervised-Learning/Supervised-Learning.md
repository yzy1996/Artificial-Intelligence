# Supervised Learning









## Optimization Problem

The goal is to find an optimal mapping function $f(x)$ to minimize the loss function of the trainig samples,
$$
\min _{\theta} \frac{1}{N} \sum_{i=1}^{N} L\left(y^{i}, f\left(x^{i}, \theta\right)\right),
$$
where $N$ is the number of training samples, $\theta$ is the parameter of the mapping function, $x^i$ is the feature vector of the $i$-th samples, $y^i$ is the corresponding label, and $L(\cdot)$ is the **loss function**. 

Regulation items are usually added to alleviate overfitting, e.g., in terms of $L_2$ norm,
$$
\min _{\theta} \frac{1}{N} \sum_{i=1}^{N} L\left(y^{i}, f\left(x^{i}, \theta\right)\right)+\lambda\|\theta\|_{2}^{2},
$$
where $\lambda$ is the compromise parameter, which can be determined through cross-validation.

