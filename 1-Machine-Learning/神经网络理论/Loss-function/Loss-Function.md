# Loss Function

https://zhuanlan.zhihu.com/p/39239829



按

- 分类损失函数
  - log loss
  - focal loss
  - KL divergence/relative entropy
  - exponential loss
  - hinge loss

- 回归损失函数
  - mean square error (MSE)
  - mean absolute error (MAE)
  - huber loss/smooth mean absolute error
  - log cosh loss
  - quantile loss











Some classical methods

- square of Euclidean distance
- cross-entropy
- contrast loss
- hinge loss
- information gain



MSE



## Object Detection





### edge and boundary detection

- Holistically-Nested Edge Detection (HED)







### Salient Object Detection

The goal is to identify the most visually distinctive objects or regions in an image and then segment them out from the background. Different from semantic segmentation, it pays more attention to very few objectives that are interesting and attractive.



Application includes image and video compression, content-aware image editing, object recognition etc.





**Dataset**

5 widely tested salient object detection benchmarks

0.08 seconds per image



#### Literature

Deeply Supervised Salient Object Detection with Short Connections





## Generative Model

- reconstruction loss $\mathcal{L}_{rec}$
- regularization loss $\mathcal{L}_{reg}$
- 



### reconstruction loss

### regularization loss

huber loss
$$
L_{\delta}(y, f(x))=
\left\{
\begin{array}{ll}
\frac{1}{2}(y-f(x))^{2} & \text { for }|y-f(x)| \leq \delta \\
\delta|y-f(x)|-\frac{1}{2} \delta^{2} & \text { otherwise }
\end{array}
\right.
$$


是为了和 MSE 和 MAE 进行比较的

MAE存在的问题是，梯度始终很大，

MSE梯度才会不断减小，但对异常值很敏感



huber loss 取长补短，但需要仔细调参数



