# Unsupervised Learning





## Optimization Problem



The optimization problem for the **$k$-means** clustering algorithm is formulated as minimizing the following loss function:
$$
\min _{S} \sum_{k=1}^{K} \sum_{x \in S_{k}}\left\|x-\mu_{k}\right\|_{2}^{2},
$$
where $K$ is the number of clusters, $x \in \mathbb{R^n}$ is the feature vector of samples, $\mu_k \in \mathbb{R^n}$ is the center of cluster $k$.



The objective of PCA is formulated to minimize the reconstruction error as
$$
\min \sum_{i=1}^{N}\left\|\bar{x}^{i}-x^{i}\right\|_{2}^{2} \quad \text { where } \quad \bar{x}^{i}=\sum_{j=1}^{D^{\prime}} z_{j}^{i} e_{j}, D \gg D^{\prime},
$$
where $N$ is the number of samples, $x_i$ is a $D$-dimensional vector, $\bar{x}^i$ is the reconstruction of $x^i$. $z^i = \{z_1^i, z_2^i, \dots, z_{D^\prime}^i\}$ is the projection of $x^i$ in $D^\prime$-dimensional coordinates. $e_j$ is the standard orthogonal basis under $D^\prime$-dimensional coordinates.



