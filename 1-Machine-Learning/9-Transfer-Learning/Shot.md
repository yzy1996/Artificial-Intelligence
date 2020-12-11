# Shot



**zero-shot**



**one-shot**

有一个训练集 $S$，有 $N$ 个类别，每个类别只有一个样本
$$
S = \{(x_1, y_1), \dots, (x_N, y_N)\}
$$
找到测试集 $\hat{x}$ 应该属于哪一类

**few-shot**



**K-shot**

有一个训练集 $S$，有 $N$ 个类别，每个类别有 $K$ 个样本





---

大体思路是：

通过一个大数据集上学到的general knowledge，学好一个 $X \rightarrow Y$ 的关系，希望能应用到其他问题上



没见过一个东西，但可以通过与他类似的特征进行判断，等以后知道这些特征属于什么类了，就可以进行判断





参考文献

第一次出现

[Zero-Shot Learning with Semantic Output Codes](http://www.cs.cmu.edu/afs/cs/project/theo-73/www/papers/zero-shot-learning.pdf)

[One-Shot Learning of Object Categories](http://vision.stanford.edu/documents/Fei-FeiFergusPerona2006.pdf)