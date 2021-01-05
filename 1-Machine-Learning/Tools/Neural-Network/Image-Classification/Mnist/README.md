English | [简体中文](./README.zh-CN.md)

<h1 align="center">Mnist handwritten digits</h1>
<div align="center">


Tutorial and Code

![python-version](https://img.shields.io/badge/python-3.6%20%7C%203.7-blue)![country](https://img.shields.io/badge/country-China-red)

</div>

## Mnist data set preparation

First download from the [website](http://yann.lecun.com/exdb/mnist/) and uncompress.

We will get four files including training set images\labels and test set images \labels.

The training set contains 60000 examples, and the test set 10000 examples.

The first 5000 examples of the test set are taken from the original NIST training set. The last 5000 are taken from the original NIST test set. The first 5000 are cleaner and easier than the last 5000.

These files are not in any standard image format. You have to write your own (very simple) program to read them.

I give you an reference [code](https://github.com/yzy1996/Artificial-Intelligence/blob/master/Machine-Learning/Image%20Classification/Mnist/data%20_extract.py).



There is a new mnist dataset offered by Facebook, [link](https://github.com/facebookresearch/qmnist) , [paper](https://arxiv.org/pdf/1905.10498.pdf)

## Using scikit-learn package

### Installing 


```
pip install scikit-learn
```

### Usage





## Without scikit-learn package







## Based on tensorflow

Use three methods to solve:

* CNN
* NN(BP)
* softmax regression

## Based on pytorch

Use three methods to solve:

* CNN
* NN(BP)
* softmax regression



ps: use tensorflow and pytorch, you need to configure your environment first, you can see [this](https://github.com/yzy1996/Artificial-Intelligence/blob/master/Machine-Learning/Configuration/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E9%85%8D%E7%BD%AE%E6%8C%87%E5%8D%97.md)