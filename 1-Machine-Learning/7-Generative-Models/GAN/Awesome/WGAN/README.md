# WGAN



## 知识点补充

什么是Wasserstein距离

[想要算一算Wasserstein距离？这里有一份PyTorch实战](https://www.jiqizhixin.com/articles/19031102)

[Read-through: Wasserstein GAN](https://www.alexirpan.com/2017/02/22/wasserstein-gan.html)



WGAN在具体实现上做了哪些改动呢？

- 损失函数里没有 $\log$ 了，$D$ 的输出层不用 sigmoid，直接使用值
- 剪裁 $D$ 的权重 （-0.01-0.01）
- 每次迭代，训练一次 $G$，多训练几次 $D$ （5次）
- 使用 RMSProp 替代 Adam
- 使用更低的学习率（$\alpha$ = 0.00005）



![Algorithm for the Wasserstein Generative Adversarial Networks](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2019/05/Algorithm-for-the-Wasserstein-Generative-Adversarial-Networks-1.png)

The differences in implementation for the WGAN are as follows:

1. No log in the loss. Use a linear activation function in the output layer of the D (instead of sigmoid).
2. Use -1 labels for real images and 1 labels for fake images (instead of 1 and 0).
3. Use Wasserstein loss to train the critic and generator models.
4. Clip the weights of D to a limited range after each mini batch update (e.g. [-0.01,0.01]).
5. Train D more than G each iteration (e.g. 5).
6. Use the RMSProp version of gradient descent with a small learning rate and no momentum (e.g. 0.00005).



Wasserstein GAN（WGAN）

- 彻底解决GAN训练不稳定的问题，不再需要小心平衡生成器和判别器的训练程度
- 基本解决了collapse mode的问题，确保了生成样本的多样性
- 训练过程中终于有一个像交叉熵、准确率这样的数值来指示训练的进程，这个数值越小代表GAN训练得越好，代表生成器产生的图像质量越高（如题图所示）
- 以上一切好处不需要精心设计的网络架构，最简单的多层全连接网络就可以做到



WGAN提出用 Wasserstein 距离取代之前的KL和JS散度
$$
W\left(P_{r}, P_{g}\right)=\inf _{\gamma \sim \prod\left(P_{r}, P_{g}\right)} \mathbb{E}_{(x, y) \sim \gamma}\|x-y\|
$$
其中 $P_r, P_g$ 分别为真实数据和生成的数据的分布函数，Wasserstein 距离衡量了这两个分布函数的差异性。直观地理解，就是根据这两个分布函数分别生成一堆数据 $x_1, x_2, \dots, x_n$ 和另一堆数据 $y_1, y_2, \dots, y_n$，然后计算这两堆数据之间的距离。距离的算法是找到一种一一对应的配对方案$\gamma \sim \prod\left(P_{r}, P_{g}\right)$ ，把 $x_i$ 移动到 $y_i$ ，求总移动距离的最小值。由于在 GAN 中， $P_r$ 和 $P_g$  都没有显式的表达式，只能是从里面不停地采样，所以不可能找到这样的 $\gamma$，无法直接优化公式 (2) 。所以作者根据 Kantorovich-Rubinstein duality，将公式 (2) 转化成公式 (3)，过程[详见](https://vincentherrmann.github.io/blog/wasserstein/)
$$
W\left(P_{r}, P_{g}\right)=\sup _{\|f\|_{L} \leq 1} \mathbb{E}_{x \sim P_{r}}[f(x)]-\mathbb{E}_{y \sim P_{g}}[f(y)]
$$
其中 $f$ 为判别器函数，只有当判别器函数满足 1-Lipschitz 约束时，(2) 才能转化为 (3)。除此之外，正如上文所说，Lipschitz continuous 的函数的梯度上界被限制，因此函数更平滑，在神经网络的优化过程中，参数变化也会更稳定，不容易出现梯度爆炸，因此Lipschitz continuity 是一个很好的性质。

为了让判别器函数满足 1-Lipschitz continuity，WGAN 和之后的 WGAN-GP 分别采用了 weight-clipping 和 gradient penalty 来约束判别器参数。



## code

$$
L_D = E[D(x)] - E[D(G(z))]
\\
L_G = E[D(G(z))]
\\
W_D \leftarrow clip_by_value(W_D, -0.01, 0.01)
$$

$$

$$

$$

$$







[官方源码](https://github.com/martinarjovsky/WassersteinGAN)



```python
def discriminator_loss_fn(real_output, fake_output):
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

def generator_loss_fn(fake_output):
    return -tf.reduce_mean(fake_output)
```


-1 for real output

1 for fake output



### Critic Weight Clipping

其他的几个改动很好修改，最大的改动就是这个权值裁剪



知道他对什么进行裁剪，对权重，还是梯度，应该是权重

对梯度进行裁剪



在tensorflow里用到 `tf.clip_by_value` 这个函数

```python
tf.clip_by_value(t, -0.01, 0.01)
```





## reading material

