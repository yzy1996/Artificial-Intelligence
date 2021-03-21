<h1 align="center">NSGA</h1>
<div align="center">

Full name is: **Non-dominated Sorting Genetic Algorithm** (非支配排序遗传算法), proposed by **Kalyanmoy Deb**

![python-version](https://img.shields.io/badge/python-3.7-blue) ![country](https://img.shields.io/badge/country-China-red)

</div>

### Main papers

NSGA-I --> NSGA-II --> NSGA-III

[Muiltiobjective Optimization Using Nondominated Sorting in Genetic Algorithms](https://ieeexplore.ieee.org/document/6791727)

[A fast and elitist multiobjective genetic algorithm: NSGA-II](http://www.dmi.unict.it/mpavone/nc-cs/materiale/NSGA-II.pdf)

[An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based Nondominated Sorting Approach, Part I: Solving Problems With Box Constraints](https://ieeexplore.ieee.org/abstract/document/6600851)

[An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point Based Nondominated Sorting Approach, Part II: Handling Constraints and Extending to an Adaptive Approach](https://ieeexplore.ieee.org/abstract/document/6595567)

### Explanation

定义一个多目标问题

### Definition

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

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/Front_pareto.svg/1280px-Front_pareto.svg.png" alt="img" style="zoom:30%;" />

进化算法在多目标优化问题上得到了很广泛的应用，通过种群的不断进化迭代，进化算法能得到一个Approximation Set，那么我们如何来评价得到的Approximation Set的优劣呢，以下为两方面的评价标准。

**收敛性(Convergence)**

Approximation Front 与 PF 的贴近程度。

**分布性(Distribution)**

描述Approximation Front 在PF 的分布情况，包括多样性(Diversity)和均匀性(Uniformity)。

具体来说，常用的两个指标分别是IGD(Inverted Generational Distance) 和 HV(Hypervolume)。其中，IGD需要知道PF数据，且其具体计算为每个PF中的点到其最近的Approximation Front中的点的距离之和的均值。同时，需注意，这两种方法都能同时度量解的分布性和收敛性。


#### Time complexity

#### 拥挤度

为了使得到的解在目标空间中更加均匀，这里引入了拥挤度

#### 精英保留策略



## 主流的多目标进化算法

- 

从多目标问题本身来说，主要分类如下：

- 基于Pareto支配关系【加入跳转本页介绍】 NSGA 【再跳转另一文档】
- 基于分解的方法
- 基于Indicator方法

先来介绍下基于遗传算法的多目标优化算法的一些基本参数：
种群大小：每次迭代都需保证种群大小是一致的，且其大小应由初始化设定。
交叉概率：用于衡量两个个体交叉的概率。
突变率：交叉产生的解发生突变的概率。
标准的遗传算法每次迭代都会将上一代的个体丢弃，虽然这符合自然规律，但对于遗传算法来说，这样效果不是特别好，因此，精英保留策略将上一代个体和当前个体混合竞争产生下一代，这种机制能较好的保留精英个体。





 改进了三个内容：(1)提出了快速非支配排序算法；(2)采用拥挤度和拥挤度比较算子；(3)引入精英策略。 





#### 实验

实验使用ZDT问题






### Code

Deb's lab code: https://www.egr.msu.edu/~kdeb/codes.shtml

other toolbox: 





### 参考

[多目标优化算法（一）NSGA-Ⅱ（NSGA2）-晓风wangchao](https://blog.csdn.net/qq_40434430/article/details/82876572)

[subtask 多目标进化算法(MOEA)概述](https://blog.csdn.net/qithon/article/details/72885053)

https://blog.csdn.net/jinjiahao5299/article/details/76045936