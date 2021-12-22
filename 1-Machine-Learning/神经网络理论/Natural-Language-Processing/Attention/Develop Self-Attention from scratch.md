# Develop Self-Attention from scratch

> To help you understand

ref 

https://jalammar.github.io/illustrated-transformer/

https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a

https://medium.com/@bgg/seq2seq-pay-attention-to-self-attention-part-2-cf81bf32c73d

[https://medium.com/@pkqiang49/%E4%B8%80%E6%96%87%E7%9C%8B%E6%87%82-attention-%E6%9C%AC%E8%B4%A8%E5%8E%9F%E7%90%86-3%E5%A4%A7%E4%BC%98%E7%82%B9-5%E5%A4%A7%E7%B1%BB%E5%9E%8B-e4fbe4b6d030](https://medium.com/@pkqiang49/一文看懂-attention-本质原理-3大优点-5大类型-e4fbe4b6d030)

This is the framework of Scaled Dot-Product Attention

![img](https://pic2.zhimg.com/80/v2-32eb6aa9e23b79784ed1ca22d3f9abf9_720w.jpg)

$$
Attention(Q, K, V) = \text{softmax}\left(\frac{QK^{\mathsf{T}}}{\sqrt{d}}\right)V
$$
QK的内积反映他们之间的相似程度，然后归一化，



## What is Query, Key and Value

可以用一个例子来理解：

如果我们要去水果一条街买水果，刚开始我们也不知道怎么买，就到每个水果摊看看，摊主就会向我们介绍水果好不好吃，产地是哪里，新不新鲜等特征，逛了足够多的水果摊了，评价水果的标准也就慢慢自发（self-attention）形成了，就知道哪些特征比较重要，哪些不重要，形成了一系列权重，然后按照这些权重去每个水果摊去买。

在这个过程中，query想买点水分多的水果；key就会有西瓜，莲雾；value是水果摊是否有这些水果。



Key1:第一个水果摊 有 Value:苹果，梨，西瓜，火龙果

Key2:第一个水果摊 有 Value:苹果，梨，西瓜，火龙果



苹果和含水量的关系





## Why use Q K V

