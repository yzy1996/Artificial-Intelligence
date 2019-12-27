<h1 align="center">GAN</h1>
<div align="center">
![Country](https://img.shields.io/badge/country-China-red)

About  Generative Adversarial Network (GAN)

</div>

## Learning path

The original paper  [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) 2014 Lan Goodfellow

存在着训练困难、生成器和判别器的loss无法只是训练进程、生成样本缺乏多样性等问题



CNN + GAN = DCGAN，依靠的是对判别器和生成器的构架进行实验枚举，最终找到一组比较好的网络架构设置。

[DCGAN paper](https://arxiv.org/abs/1511.06434)

[DCGAN-tensorflow code](https://github.com/carpedm20/DCGAN-tensorflow)



[GAN学习指南：从原理入门到制作生成Demo](https://zhuanlan.zhihu.com/p/24767059)



Wasserstein GAN（WGAN）

- 彻底解决GAN训练不稳定的问题，不再需要小心平衡生成器和判别器的训练程度
- 基本解决了collapse mode的问题，确保了生成样本的多样性
- 训练过程中终于有一个像交叉熵、准确率这样的数值来指示训练的进程，这个数值越小代表GAN训练得越好，代表生成器产生的图像质量越高（如题图所示）
- 以上一切好处不需要精心设计的网络架构，最简单的多层全连接网络就可以做到

[paper](https://arxiv.org/abs/1701.07875)

[code](https://github.com/martinarjovsky/WassersteinGAN)

?? What is Wasserstein. 

[Read-through: Wasserstein GAN](https://www.alexirpan.com/2017/02/22/wasserstein-gan.html)

[想要算一算Wasserstein距离？这里有一份PyTorch实战](https://www.jiqizhixin.com/articles/19031102)