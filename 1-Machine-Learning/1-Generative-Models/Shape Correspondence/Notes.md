





为了构建一个 flow + confidence 的统一模型



为什么这样统一了就好：因为都有需求



definition



a given image pair $X = (I^q, I^r)$ of spatial size $H \times W$, the aim of dense matching is to estimate a flow field $Y \in \mathbb{R}^{H \times W \times 2}$. 



Most learing-based methods address this problem by training a network $F$ with parameters $\theta$ that directly predicts the flow as $Y=F(X;\theta)$.



This work additionally learn the conditional probability density $p(Y|X;\theta)$



in Flow cases, there is a commonly performed method



不确定性预测：uncertainty prediction 





Global correlation layer
$$
C_G(f^r, f^q)_{ijkl} = (f_{ij}^r)^\mathsf{T}f_{ij}^q, (i,j),(k,l) \in \{1, \dots, H\} \times \{1, \dots, W\}
$$
这是一个4D的张量

Local correlation layer，就仅仅对（i,j）的邻域去scalar product
$$
C_L(f^r, f^q)_{ijkl} = (f_{ij}^r)^\mathsf{T}f_{ij}^q, (i,j)\in \{1, \dots, H\} \times \{1, \dots, W\}, (k,l) \in \{-R,\dots, R\}
$$







$$
w^* = \arg\min_w  
$$






