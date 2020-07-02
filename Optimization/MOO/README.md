# Multi-Objective Optimization Survey

two survey of this field:

- Nonlinear Multiobjective Optimization by Miettinen 1998
- Multicriteria Optimization by Ehrgott 2005



### 1. Gradient based multi-objective optimization

The development of multiple gradient descent algorithm (MGDA) can be summarized as:

[Steepest descent methods for multicriteria optimization](./Steepest-descen-methods-for-multicriteria-optimization.pdf) by Fliege (2000)

<details><summary>Click to expand</summary><p>


**The main work:**

> They propose two parameter-free optimization methods for computing a point satisfying first-order necessary conditions for multicriteria optimization. Neither ordering information nor weighting factors for the different objective functions is assumed to be known.

They formulate a unconstrained minimization problem:

$$
\begin{aligned} 
\min \quad & f_{x}(v)+\frac{1}{2}\|v\|^{2} \\
\text{subject to} \quad & v \in \mathbb{R}^{n}
\end{aligned}
$$

Since the objective function is proper, closed, and strongly convex, it has always a (unique) solution. Note that a simple reformulation to get rid of the non-differentiabilities would be

$$
\begin{aligned} 
\min \quad & \alpha + \frac{1}{2}\|v\|^{2} \\
\text{subject to} \quad & (Av)_i  \leq \alpha, \quad i=1, \ldots, m
\end{aligned}
$$

Of course, there is no need for the specific choice of $(1/2)\|\cdot\|^2$ as the function. In fact, any proper closed strictly convex function can be used.

</p></details>

---

[Stochastic method for the solution of unconstrained vector optimization problems](./Stochastic-method-for-the-solution-of-unconstrained-vector-optimization-problems.pdf) by Schäffler (2002)

<details><summary>Click to expand</summary><p>


**The main algorithm:**

> They propose a new stochastic algorithm for the solution of unconstrained vector optimization problems, which is based on a special class of stochastic differential equations.

$$
\text{(QOP(X))} \quad \min _{\alpha \in \mathbb{R}^{m}}\left\{\left\|\sum_{i=1}^{m} \alpha_{i} \nabla f_{i}(x)\right\|_{2}^{2}, \alpha_{i} \geq 0, i=1, \ldots, m, \sum_{i=1}^{m} \alpha_{i}=1\right\}
$$

where $f = (f_1, f_2, \dots, f_m)^T \quad f_i:\mathbb{R}^{n} \rightarrow \mathbb{R} \quad i=1, \dots, m.$

There are some properties of the problem resulting from convex analysis:

(i) For each $x \in \mathbb{R}^n$, there exits a global minimizer $\hat{\alpha}$ of (QOP(X)), which is not unique in general. Each local minimizer of (QOP(X)) is a global minimizer.

(ii) Let $\hat{\alpha}$ and $\tilde{\alpha}$ be two global minimizers of (QOP(X)) for fixed $x \in \mathbb{R}^n$. Then,

$$
\sum_{i=1}^{m} \hat{\alpha}_{i} \nabla f_{i}(x)=\sum_{i=1}^{m} \tilde{\alpha}_{i} \nabla f_{i}(x)
$$

</p></details>

---

[Multiple-gradient descent algorithm (MGDA) for multiobjective optimization](./Multiple-gradient-descent-algorithm(MGDA)-for-multiobjective-optimization.pdf) by Désidéri (2012)

<details><summary>Click to expand</summary><p>


**The main algorithm:**

> 

**Methods it used:** 

- [ ] 

**Its contribution:**

> 

**My Comments:**

> 
>

</p></details>

---


All of the above methods use multi-objective Karush-Kuhn-Tucker (**KKT**) conditions and find a descent direction that decreases all objectives. This approach was extended to **stochastic gradient descent**:



[Descent algorithm for nonsmooth stochastic multiobjective optimization](./Descent-algorithm-for-nonsmooth-stochastic-multiobjective-optimization.pdf) by Poirion (2017)

<details><summary>Click to expand</summary><p>


**The main work:**

> 

**Methods it used:** 

- [ ] 

**Its contribution:**

> 

**My Comments:**

> 
>

</p></details>

---

[Gradient-Based Multiobjective Optimization with Uncertainties](./Gradient-Based-Multiobjective-Optimization-with-Uncertainties.pdf) by Peitz (2018)

<details><summary>Click to expand</summary><p>


**The main work:**

> They develop a gradient-based algorithm for the solution of multiobjective optimization
> problems with uncertainties. Uncertainties mean inexact gradients.

</p></details>

---
