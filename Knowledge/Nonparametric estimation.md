Nonparametric estimation

如果有关于模型的知识，就可以用参数估计；而如果没有，就要用到非参数估计



Histogram 直方图



我们有 samples ${x_1,x_2,x_3,\dots,x_N}$

consider region R,

* define 概率 



概率密度的估计 $p(x)= \frac{K}{NV}$

K是位于区域R内部的数据点个数，N是数据点总数 ，V是区域R的体积



固定K，调整V，就是K近邻

固定V，调整K，就是核方法



我们现在构造一个以原点为中心的单位立方体，如果数据点 $x_n$ 位于以x为中心的边长为h的立方体中， $k(\frac{x-x_n}{h})$ 的值等于1，否则等于0，于是位于这个立方体内的数据点总数为
$$
K=\sum_{n=1}^{N} k\left(\frac{\boldsymbol{x}-\boldsymbol{x}_{n}}{h}\right)
$$

$$
p(\boldsymbol{x})=\frac{1}{N} \sum_{n=1}^{N} \frac{1}{h^{D}} k\left(\frac{\boldsymbol{x}-\boldsymbol{x}_{n}}{h}\right)
$$

