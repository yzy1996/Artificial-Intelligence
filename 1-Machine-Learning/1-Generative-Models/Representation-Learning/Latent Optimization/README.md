# Latent Optimization







the data transformations are defined using linear paths in a Euclidean latent space



先描述问题，再找解决办法



Learning transport operators for image manifolds



> we desire a manifold model that allows us to learn the data structure, generate points outside of the training set, and map out smooth manifold paths through the latent space.



Decoder或者Generator这样的是从一个低维到高维的映射，如果这个分布不能匹配源数据的分布，生成的效果就会很差。

learning mappings from low-dimensional latent vectors to high dimensional data while 



According to the manifold hypothesis, high-dimensional data can often be modeled as lying on or near a low-dimensional, nonlinear manifold

Testing the manifold hypothesis





[Variational Autoencoder with Learned Latent Structure](https://arxiv.org/pdf/2006.10597.pdf)  
*Marissa C. Connor, Gregory H. Canal, Christopher J. Rozell*  
**[`AISTATS 2021`] ()**

> 用VAE的后验衡量向量z落在embedding领域流型上的概率；这个流型是被teansport operator学到的。
>
> 通过网络 $f\phi(\cdot)$ 编码 latent coordinates of x ? （后面这句如何去理解呢）
>
> 然后从 $q_\phi(z \mid x)$ 中采样。

![image-20220227172830807](https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/image-20220227172830807.png)



The anchor 是指的什么呢？

以及如何在实验中定义一个Ground truth latent structure
