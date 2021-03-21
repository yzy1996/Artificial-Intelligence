# MGDA Survey

（首先我们需要一句定性的描述）

The Multi-Gradient Descent Algorithm (MGDA) is an extension of the classical Gradient Descent Algorithm to multiple objectives. This algorithm has been proved to converge to the Pareto Stationary solution.

（其次我们要介绍他的历史）

The Multiple Gradient Descent Algorithm ( MGDA ) was originally introduced by [Desideri (2009)](Multiple-gradient descent algorithm) and [Desideri (2012)](Multiple-gradient descent algorithm ( MGDA ) for multiobjective optimization) to solve multi-objective optimization problems involving differentiable objective functions.

(2014). Multiple-gradient descent algorithm ( MGDA ) for Pareto-front identification

(2015). Révision de l’algorithme de descente àgradients multiples

(2018). Quasi-Riemannian Multiple Gradient Descent Algorithm for constrained multiobjective differential optimization

(2017). Parametric optimization of pulsating jets in unsteady flow by multiple-gradient descent algorithm



## Definition

MGD finds a **common descent directiion** for all $f_i$ by defining the convex hull of all $\nabla f_{i}(\mathbf{x})$ and finding the minimum norm element within it. 
$$
\min _{\lambda_{1}, \ldots, \lambda_{m}}\left\{\left\|\sum_{i=1}^{m} \lambda_{i} \nabla \mathcal{L}_{i}(\theta)\right\| \mid \sum_{i=1}^{m} \lambda_{i}=1, \lambda_{i} \geq 0\right\}
$$

where $\nabla \mathcal{L}(\theta) = \sum_{i=1}^{m} \lambda_{i} \nabla \mathcal{L}_{i}(\theta)$ is the descent direction. MGD then update parameters with a learning rate $\lambda$ according to $\mathbf{x}_{t+1}=\mathbf{x}_{t}-\lambda \nabla f(\mathbf{x})$.



To solve easily, the definition can be reformulated as a Quadratic Constrained Optimization Problem (QCQP). The QCQP is defined as follows:
$$
\min _{\lambda_{1}, \ldots, \lambda_{m}}\left\{\left\|\sum_{i=1}^{m} \lambda_{i} \nabla \mathcal{L}_{i}(\theta)\right\|^2 \mid \sum_{i=1}^{m} \lambda_{i}=1, \lambda_{i} \geq 0\right\}
$$
Besides, based on the calculated value, we can either get a stop condition:

- $\nabla \mathcal{L}(\theta)=0$, the solution is Pareto Stationary;
- $\nabla f(\mathbf{x}) \neq 0$, the solution is not Pareto Stationary.



Another expression of search direction $d^k$ is:
$$
d^{k}=\arg \min _{d \in \mathbb{R}^{n}}\left\{\max _{i \in\{1, \ldots, m\}} \nabla f_{i}\left(x_{k}\right)^{\top} d+\frac{1}{2}\|d\|^{2}\right\}
$$
This can be also rewritten equivalently as the following quadratic optimization problem:
$$
\begin{equation}
\left(d^{k}, \alpha^{k}\right)=\arg \min _{d \in \mathbb{R}^{n}, \alpha \in \mathbb{R}} \alpha+\frac{1}{2}\|d\|^{2}
\\
\text { subject to } \nabla f_{i}\left(x^{k}\right)^{\top} d \leq \alpha, \quad i=1, \ldots, m
\end{equation}
$$

- If $x^k$ is Pareto critical, then $d^k = 0 \in \mathbb{R}^{n}$ and $\alpha^k = 0$





Based on the number of objectives, there are two different ways of how this QCQP can be solved: with an analytical solution for two objectives or with a constrained optimization problem for more objectives.

### Two Objectives

In case of two objectives 
$$
\min \|\lambda * \nabla f_1 + (1 - \lambda) * \nabla f_2\|^2
$$
an analytical solution of this problem is:
$$
\lambda = \frac{(\nabla f_2-\nabla f_1)^{T} * \nabla f_2}{\|\nabla f_1-\nabla f_2\|^{2}}
$$
where $\lambda$ is clipped to $[0, 1]$



### Multiple Objectives

Now we cannot compute an exact solution 



The efficient solution that scales nicely to the high-dimensional problems is proposed in ([Sener and Koltun, 2018](Multi-task learning as multi-objective optimization)). The proposed solution is based on Frank-Wolfe constrained optimization algorithm ([Frank and Wolfe, 1956](An algorithm for quadratic programming)). 



Using experiment to present the **efficient**, **training time** and **convergence properties** of the solver.





## Shortcoming of MGDA

- Calculating full gradient at every optimization step is computationally expensive
- As a [deterministic optimization algorithm](), it can quickly become stuck at a bad Pareto stationary point; the same way as a full gradient descent algorithm can quickly become stuck at the bad local minimum.
- The requirement of calculating gradient for the objective function restricts from using non-smooth loss functions as objective functions (e.g. Mean Absolute Error (MAE))



## Solving the drawbacks

([Poirion, 2017]( Descent algorithm for nonsmooth stochastic multiobjective optimization)) propose an extension of MGDA to address its limitation, called Stochastic Multi Subgradient Descent Algorithm (SMSGDA). （说优点）computational cost, stuck at local stationary point and non-smooth loss function.

However stochastic means that the KKT condition could not be satisfied as we do not calculate full gradient. We can adjust the stop conditions of the optimization process: (i) the number of epochs; (ii) the loss of the common descent is plateauing; (iii) the gradient norm is less than $\epsilon$.



## Gradient Normalization

Different objectives have different scales. Both MGDA and SMSGDA are sensitive to the value scales. This will cause that one single objective may completely dominates the whole optimization process. To solve this problem, [(Nikola, 2020)](Multi-Gradient Descent for Multi-Objective Recommender Systems) propose a gradient normalization method:
$$
\nabla_{\theta} \hat{\mathcal{L}}_{i}(\theta)=\frac{\nabla_{\theta} \mathcal{L}_{i}(\theta)}{\mathcal{L}_{i}\left(\theta_{i n i t}\right)}
$$
where $\nabla_{w} \hat{L}_{i}(w)$ is the normalized gradient, $\nabla_{w} L_{i}(w)$ the non-normalized gradient, $L_{i}\left(w_{i n i t}\right)$ the initial loss of the objective. They consider $L_{i}\left(w_{i n i t}\right)$ to be an  empirical maximum loss for the particular objective.