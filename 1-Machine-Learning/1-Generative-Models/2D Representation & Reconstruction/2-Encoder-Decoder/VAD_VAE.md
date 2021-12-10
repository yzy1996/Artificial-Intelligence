# Variational Auto-Encoder (VAE)

有一个mean和一个log_var



当目标分布是 $\mathcal{N}(0,1^2)$ 时，mean=0, log_var=0

当目标分布是 $\mathcal{N}(0,0.01^2)$ 时，mean=0, log_var=-9

而当mean和log_var也是分布的时候，比如初始化都是 $\mathcal{N}(0, 1^2)$

就应该分别变化到，$\mathcal{N}(0, 0.01^2)$ $\mathcal{N}(-9, 0.01^2)$ 波动小比较好
