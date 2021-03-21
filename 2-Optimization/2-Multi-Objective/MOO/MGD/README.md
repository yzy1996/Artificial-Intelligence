# Multiple Gradient Descent (MGD)

Multiple gradient descent ([Schäffler, 2002](Stochastic method for the solution of unconstrained vector optimization problems), [Désidéri, 2012](Multiple-gradient descent algorithm (MGDA) for multiobjective optimization), [Peitz, 2018](Gradient-based multiobjective optimization with uncertainties)) was proposed for the unconstrained case of multi-objective optimization of $\mathbf{F}(\mathbf{x})$ assuming a **convex**, **continuously differentiable** and **smooth** $f_k(x)$ for all $k$.



MGD finds a common descent direction for all $f_k$ by defining the convex hull of all $\nabla f_{k}(\mathbf{x})$ and finding the minimum norm element within it. Consider $\mathbf{w}^{*}$ given by:

$$
\begin{equation}
\mathbf{w}^{*}=\operatorname{argmin}\|\mathbf{w}\|, \quad \mathbf{w}=\sum_{k=1}^{K} \alpha_{k} \nabla f_{k}(\mathbf{x}) \\
\text { s.t. } \quad \sum_{k=1}^{K} \alpha_{k}=1, \quad \alpha_{k} \geq 0 \quad \forall k
\end{equation}
$$
where $w^*$ is a descent direction for all $f_i(\mathbf{x})$ . MGD then update parameters with a learning rate $\lambda$ according to $\mathbf{x}_{t+1}=\mathbf{x}_{t}-\lambda \frac{w_{t}^{*}}{\left\|\mathbf{w}_{t}^{*}\right\|}$ 



## Stochastic Multi-Gradient (SMG)

which can be seen as an extension of the classical stochastic gradient method for single-objective optimization. 









## Reference

[1]The stochastic multi-gradient algorithm for multi-objective optimization and its application to supervised machine learning










