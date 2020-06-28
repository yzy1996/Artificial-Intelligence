<h1 align="center">Optimization</h1>
<div align="center">
Download and view in Typora

![country](https://img.shields.io/badge/country-China-red)

</div>

## Multi objective optimization

> In multi-objective optimization, several objective functions have to be minimized simultaneously. Usually, no single point will minimize all given objective functions at once, and so the concept of optimality has to be replaced by the concept of Pareto optimality. 

> A point is called **Pareto-optimal** or efficient, if there does not exist a different point with the same or smaller objective function values, such that there is a decrease in at least one objective function value.

> Each **local Pareto optimal** point is **globally Pareto optimal** as soon as all functions Fi (i = 1, . . . ,m) are convex.

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/Front_pareto.svg/1280px-Front_pareto.svg.png" alt="img" width = "300" align = "center" />

1. **Pareto dominance** （支配）

   目标函数表示为 $\underset{x}{min}\ F(x)=(F_1(x), F_2(x),\dots, F_m(x))$ ，若目标函数的两个解 $x_1$, $x_2$ 满足:
   $$
   \left\{
   \begin{array}{l} 
   F_i(x_1) \leq F_i(x_2),\ \forall i \in\{1, \ldots, m\}\\
   F_i(x_1) < F_i(x_2),\ \exists i \in\{1, \ldots, m\}
   \end{array}
   \right.
   $$
   就称解 $x_1$ 支配 $x_2$

2. **Pareto optimality**（最优）

   如果存在解 $x^*$ ，没有其他能支配 $x^*$ 的解，就称解 $x^*$ 是帕累托最优解。所有的帕累托最优解构成了帕累托前沿。

3. **Pareto Set** （集）

   如果一组给定的最优解集中的解是相互非支配的，就称这个解集为帕累托集
   
4. **Pareto Front** （前沿）

   帕累托集中每个解对应的目标值向量组成的集合称为帕累托前沿

5. **Approximation Set** （近似集）

   准确的帕累托集很难获得，

6. **Approximation Front** （近似前沿）

   类似帕累托前沿，近似集产生的是近似前沿



constrained multicriteria optimization problem

unconstrained multicriteria optimization problem



Organization of the learning progress

### Development

| year |      event      | related papers |
| :--: | :-------------: | :------------: |
| 1906 | Birth of Pareto |                |
| 1979 |                 |                |
| 2002 |     NSGA-II     |                |
| 2014 |                 |                |
|      |                 |                |

The first generation

1.1 MOGA，提出用进化算法来解决多目标问题

1.2 Niched Pareto Genetic Algorithm, NPGA 

1.3 Non-dominated Sorting Genetic Algorithm, NSGA

The second generation

2.1 Strength Pareto Evolutionary Algorithm SPEA SPEA2

2.2 Non-dominated Sorting Genetic Algorithm 2 NSGA2

2.3 Pareto Archived Evolution Strategy PAES 

### Main method

add a sort and direct to subsection(introduction and code)

*  Weighted Sum Method 
*  ε-constraint method 
*  Multi-Objective Genetic Algorithms 



### Code

[python-geatpy](http://geatpy.com/)

[matlab-gamultiobj](https://ww2.mathworks.cn/help/gads/gamultiobj.html)

[Evolutionary multi-objective optimization platform](https://github.com/BIMK/PlatEMO)

### papers

Multi-objective optimization using genetic algorithms: A tutorial [2006]

Multi-objective Optimization Using Evolutionary Algorithms: An Introduction [2011]

### Application

 [多目标遗传算法NSGA-Ⅱ与其Python实现多目标投资组合优化问题](https://blog.csdn.net/WFRainn/article/details/83753615) 



https://blog.csdn.net/quinn1994/article/details/80679528







## Multi-objective Evolutionary Algorithms (MOEAs)
- the dominance based algorithms (NSGA, SPEA)
- decomposition-based MOEAs (MOEA/D, MOEA/D-DE)
- the performance indicator based algorithms (SMS-EMOA)



the third generation differential algorithm (GDE3)

the memetic Pareto achieved evolution strategy (M-PAES)



A general MOEA framework

offspring reproduction, fitness assignment, environmental selection