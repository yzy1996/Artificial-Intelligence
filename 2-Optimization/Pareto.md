<h1 align="center">Pareto</h1>
<div align="center">
Download and view in Typora


![country](https://img.shields.io/badge/country-China-red)

</div>

## Pareto

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

3. **Pareto-stationarity**  （稳定）

   帕累托稳定是帕累托最优的必要条件。若存在一组权重 $\alpha_k$， 满足：

   $$
   \sum_{k=1}^{K} \alpha_{k} \nabla f_{k}=\mathbf{0}, \quad \sum_{k=1}^{K} \alpha_{k}=1, \quad \alpha_{k} \geq 0 \quad \forall k
   $$
   就称 $F$ 在 $x$ 达到了帕累托稳定

   多维共面（二维共线），

4. **Pareto Set** （集）

   如果一组给定的最优解集中的解是相互非支配的，就称这个解集为帕累托集

5. **Pareto Front** （前沿）

   帕累托集中每个解对应的目标值向量组成的集合称为帕累托前沿

6. **Approximation Set** （近似集）

   准确的帕累托集很难获得，

7. **Approximation Front** （近似前沿）

   类似帕累托前沿，近似集产生的是近似前沿


