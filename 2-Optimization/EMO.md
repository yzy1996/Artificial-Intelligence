<h1 align="center">Evolutionary Multi-objective Optimization (EMO)</h1>
<div align="center">
     EMO is a population based heuristic stochastic search method  
</div>



**First, we will start with several questions to ask ourselves**

(i) Why do we find the Pareto-front?

(ii) Can we find an analytical solution of Pareto-front?

(iii) Does the discrete solution approximate the complete Pareto-front?



Some basic knowledge is [here](./Basic-Knowledge.md) and the general form of the multi-objective optimization problem can be stated as:
$$
\begin{aligned}
\text { Minimize/Maximize } & f_{m}(\mathrm{x}), & m=1,2, \ldots, M ; \\
\text { subject to } & g_{j}(\mathrm{x}) \geq 0, & j=1,2, \ldots, J ; \\
& h_{k}(\mathrm{x})=0, & k=1,2, \ldots, K ; \\
& x_{i}^{(L)} \leq x_{i} \leq x_{i}^{(U)}, & i=1,2, \ldots, n
\end{aligned}
$$
A solution $x \in R^n$ is expressed as $\mathbf{x} = (x_1, x_2, \dots, x_n)^T$ and its corresponding objective space is denoted by $f(\mathbf{x} = \mathbf{z} = (z_1, z_2, \dots, z_M)^T$. 



The concept of the term *domination* is summarized in [here](./NSGA/README.md)



**The goals of multi-objective optimization:**

(i) find a set of solutions which lie on the Pareto-optimal front

(ii) Find a set of solutions which are diverse enough to represent the entire range of the Pareto-optimal front.



**The goals of an EMO procedure**:

(i) a good convergence to the Pareto-optimal front

(ii) a good diversity in obtained solutions

> Pareto-front is the most important part of EMO and the result of population evolving is some discrete points. These points will be used to approximate a Pareto-front and so we hope these points or trade-off solutions could be as evenly distributed as possible along the ideal Pareto-front.



**How to measure the performance of EMO:**

(i) Metrics evaluating convergence to the known Pareto-optimal front (error ratio, distance from
reference set)

(ii) Metrics evaluating spread of solutions on the known Pareto-optimal front (spread, spacing)

(iii) Metrics evaluating certain combinations of convergence and spread of solutions (hypervolume, coverage, R-metrics)



**To answer the question (i):**

A lack of knowledge of good trade-off regions before a decision is made may allow the decision maker to settle for a solution which, although optimal, may not be a good **compromised solution**. The Pareto-front can help narrow down the choices and allow a decision maker to make a better decision. Meanwhile, the EMO is able to pick a particular region for **further analysis** or a particular solution for implementation.



**Supplement**

(i) The computational effort needed to select the points of the non-domination front from a set of $N$ points is $O(N \log N)$ for 2 and 3 objectives, and $O(N \log ^{M-2} N)$ for $M > 3$ objectives



Reference

[1] [Multi-Objective Optimization Using Evolutionary Algorithms](./Literature/Multi-Objective Optimization Using Evolutionary Algorithms.pdf)