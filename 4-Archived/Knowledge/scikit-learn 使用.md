# scikit-learn 使用

## 交叉验证

有两种方式

划分成两个数据集再处理

`sklearn.model_selection.train_test_split`

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)
```



自动的Model selection

 [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation)

Greadsearch





## [Clustering](https://scikit-learn.org/stable/modules/clustering.html)

提供了主流的很多算法，并有比较

