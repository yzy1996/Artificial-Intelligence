<h1 align="center">Optimization</h1>
<div align="center">



 Introduction  

![country](https://img.shields.io/badge/country-China-red)

</div>

## Multi objective optimization

In multi-objective optimization, several objective functions have to be minimized simultaneously. Usually, no single point will minimize all given objective functions at once, and so the concept of optimality has to be replaced by the concept of Pareto optimality. <!--hh-->

~111~  ^555^   

这是第一个有注脚的文本。^[注脚内容 第一条]

Organization of the learning progress

[^ss]: s



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