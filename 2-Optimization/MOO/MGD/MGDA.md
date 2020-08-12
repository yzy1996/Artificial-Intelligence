# MGDA Survey

The Multi-Gradient Descent Algorithm (MGDA) is an extension of the classical Gradient Descent Algorithm to multiple objectives.



The Multiple Gradient Descent Algorithm ( MGDA ) was originally introduced by [Desideri (2009)]() and [Desideri (2012)]() to solve multi-objective optimization problems involving differentiable objective functions.

`Multiple-gradient descent algorithm (MGDA) for multiobjective optimization`

(2009). Multiple-gradient descent algorithm

(2012). Multiple-gradient descent algorithm ( MGDA ) for multiobjective optimization

(2014). Multiple-gradient descent algorithm ( MGDA ) for Pareto-front identification

(2015). Révision de l’algorithme de descente àgradients multiples

(2018). Quasi-Riemannian Multiple Gradient Descent Algorithm for constrained multiobjective differential optimization

2017. Parametric optimization of pulsating jets in unsteady flow by multiple-gradient descent algorithm

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