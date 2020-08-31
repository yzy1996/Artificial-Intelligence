# 超参数优化

常用的超参数调参的方法有：网格搜索，随机搜索与贝叶斯优化



## 网格搜索











## 工具箱

**lightgbm**：微软开源，A fast, distributed, high performance gradient boosting (GBT, GBDT, GBRT, GBM or MART) framework based on decision tree algorithms, used for ranking, classification and many other machine learning tasks.

[github](https://github.com/Microsoft/LightGBM)

```python
from sklearn.model_selection import GridSearchCV
```

```python
from sklearn.model_selection import  RandomizedSearchCV
```

BayesOpt包来进行贝叶斯优化调参