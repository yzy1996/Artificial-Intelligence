Humans possess an impressive intuition for 3D shapes: given partial observations of an object we can easily imagine the shape of the complete object. Now we want to reproduce this ability with algorithms.







[MetaSDF: Meta-learning Signed Distance Functions](https://arxiv.org/pdf/2006.09662.pdf)

**[`NeurIPS 2020`]**	**(`Stanford`)**

**[`Vincent Sitzmann`, `Eric R. Chan`, `Richard Tucker`, `Noah Snavely`, `Gordon Wetzstein`]**

a dataset $\mathcal{D}$ of $N$ shapes, each shape is represented by a set of points $X_i$ consisting of $K$ point samples


$$
\mathcal{D} = \{X_i\}_{i=1}^N, \qquad X_i = \{(\mathbf{x}_j, s_j):s_j = SDF_i(\mathbf{x}_j)\}_{j=1}^K
$$
where $\mathbf{x}_j$ are spatial coordinates, and $s_j$ are the signed distances at these spatial coordinates