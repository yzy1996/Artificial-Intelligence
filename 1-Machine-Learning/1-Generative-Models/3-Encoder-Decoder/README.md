# 对比 AutoEncoder 和 AutoDecoder



首先分别介绍两者结构，以下将用AE和AD分别指代全称

## AutoEncoder

![AutoEncoder](https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210329100645.svg)



论文：

Reducing the Dimensionality of Data with Neural Networks *Hinton et al*.



训练过程是：

推断过程是：

采样过程？是随机像GAN那样采样还是从已经训练过的数据得到的z里面去组合采样



任务目标：

- 降维 dimension reduction

这个其实很好理解，中间的 bottleneck 隐变量 就是去掉了不重要的，留下最重要的表征，有点像PCA或者MF (Matrix Factorization)。

- 生成

## AutoDecoder

![AutoDecoder](https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20210329100810.svg)

训练过程是：

推断过程是：



## 优劣比较

- AD可以做到增量学习

知识库可以做到更新，训练数据不固定





> 思考？我们人的认知过程是AE还是AD呢？



黑夜中看到一个物体的棱角，我们是不会去想办法表征的，而是会去猜测，然后一步步走进加强（改进）这个猜测，而这里能够改变的就是z，而不是Decoder