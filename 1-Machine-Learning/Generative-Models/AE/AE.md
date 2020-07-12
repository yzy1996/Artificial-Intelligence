# Autoencoder(AE)

> reduces data dimensions by learning how to ignore the noise in the data
>
> 目前自编码器的应用主要有两个方面，第一是数据去噪，第二是为进行可视化而降维。配合适当的维度和稀疏约束，自编码器可以学习到比PCA等技术更有意思的数据投影。
>
> 对于2D的数据可视化，[t-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)（读作tee-snee）或许是目前最好的算法，但通常还是需要原数据的维度相对低一些。所以，可视化高维数据的一个好办法是首先使用自编码器将维度降低到较低的水平（如32维），然后再使用t-SNE将其投影在2D平面上
>
> 自编码器并不是一个真正的无监督学习的算法，而是一个自监督的算法。自监督学习是监督学习的一个实例，其标签产生自输入数据。

![img](https://miro.medium.com/max/700/1*P7aFcjaMGLwzTvjW3sD-5Q.jpeg)

Playing Atari with DRL



find the function that maps $x$ to $x$, Mathematically,
$$
z=f(h_e(x))
$$

$$
\hat{x}=f(h_d(z))
$$









## Denoising Autoencoder(DAE)

参考 https://blog.keras.io/building-autoencoders-in-keras.html
https://www.tensorflow.org/tutorials/generative/cvae