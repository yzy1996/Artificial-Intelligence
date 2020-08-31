博弈是怎么体现的呢，竞争中通常是让自己变好的同时让对手变差，所以双方都在为了让自己变好，让对手变差，就成了博弈



## 什么是GAN？

### 从名字解释

G代表什么

> 目的是为了得到一组合成的数据
>
>  The purpose or the result is to get new, synthetic instances of data

A代表什么

> 两个组成部分-生成器和判别器，生成器想让判别器判断自己生成的假数据是真数据，判别器想分辨出真数据和假数据，因此这是一个对抗的过程，最终的结果是两者达到一个平衡，生成器能够生成足以以假乱真的图片
>
> Two part Generator 

N代表什么

> 生成器和判别器是用神经网络来表示的
>
> **G**enerators and **D**iscriminators are represented by neural networks



### 整体是个什么流程呢？

（用那个结构图）



训练过程：**生成器**和**判别器**的博弈 （生成器生成假数据，判别器将假数据和真数据区分开；生成器重新生成假数据，判别器再次将他们区分开…周而复始）

测试过程：**生成器**生成足以以假乱真的数据



### 换一些其他的理解：

生成器和判别器分别学到了什么，首先两者学到的都是一种映射关系，判别器学习将特征映射到标签，而生成器学习将标签映射到特征

Discriminator map features to labels

Generator map labels to features



Discriminative models learn the boundary between classes

Generative models model the distribution of individual classes



## GAN为什么重要

**名人有点评**

Yann LeCun: GAN is the most interesting idea in the last 10 years in ML 



**有哪些应用**

扩充数据集，编辑图片



**什么是最有价值的呢？**

学习表征。GAN刚发展的时候，只需要他相似，但具体怎么相似的就不管了，但其实也可以管怎么相似的。传统分类器只是分类，学到的特征很局限；生成器学到的特征更好。GAN的生成可以建立对场景的语义理解，因此在这个基础上可以在语义层次做操作，比如用训练好的生成器将图片映射到潜在空间上去，能够更好得分类。



**生成的假的图片有什么用呢**？

- 可以类比黑客的价值，推动技术向更好更稳定的方向发展

robust

- 足以以假乱真，这本身就是一个很有趣的工作

字写得很好看的，画画画得很好的，一方面开创者是创造，另一方面还有临摹者，难道他们就没价值吗



## 为什么要引入CGAN？





## 什么是CGAN？





解耦操作



earn disentangled representations in a completely unsupervised manner.

disentangles writing styles from digit shapes on the MNIST dataset