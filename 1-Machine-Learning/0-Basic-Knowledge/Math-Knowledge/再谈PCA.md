最近读到ICLR2021上一篇新的论文-[EigenGame: PCA as a Nash Equilibrium](https://arxiv.org/pdf/2010.00554.pdf) (来自Deepmind)



The principal components (PCs) of data are the vectors that align with the directions of maximum variance. These have two main purposes: 

- as interpretable features 
- for data compression



主要亮点在于从一个新颖的角度看待了$k$-PCA，传统方法都是从优化的角度找出这k个主向量，而这篇文章将他们看成玩家，最终要达到一个纳什平衡

