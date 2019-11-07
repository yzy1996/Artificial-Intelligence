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

I give you an reference [code]().

## Using scikit-learn package

### Installing 


```
pip install scikit-learn
```

### Usage





## Without scikit-learn package