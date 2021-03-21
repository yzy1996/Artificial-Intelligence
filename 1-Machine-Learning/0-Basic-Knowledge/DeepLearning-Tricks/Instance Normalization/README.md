# Instance Normalization

described in the paper [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)



The principle is as following:
$$
y=\frac{x-\mathrm{E}[x]}{\sqrt{\operatorname{Var}[x]+\epsilon}} * \gamma+\beta
$$




and pytorch build in this function

```python
m = nn.InstanceNorm2d(100)
```





# Literature

[Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/pdf/1607.08022.pdf)

**[`Arxiv 2016`]**	()

**[`Dmitry Ulyanov`, `Andrea Vedaldi`, `Victor Lempitsky`]**





[link](https://github.com/aleju/papers/blob/master/neural-nets/Instance_Normalization_The_Missing_Ingredient_for_Fast_Stylization.md)