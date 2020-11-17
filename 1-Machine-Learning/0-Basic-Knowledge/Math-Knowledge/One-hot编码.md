# One-hot encoding

译名：独热编码|一位有效编码

>官方定义(from [Wikipedia](https://en.wikipedia.org/wiki/One-hot)): 当编码成一个序列后，只有一个元素为1，其余元素全为0，就被称为**one-hot**编码；相应的，若只有一个元素为0，就被称为**one-cold**编码。



### 举例

> example1 : 若一个字典里有N个单字，则每个单字可以被一个N维的one-hot向量表示。例如字典里有apple, banana, orange这三个单字，则他们各自的one-hot向量为：
>
> apple = [1 0 0]
>
> banana = [0 1 0]
>
> orange = [0 0 1]

---

> example2 : 若对多个特征进行编码，则依次将每个特征的one-hot编码拼接起来。例如有下面三个特征，`["male", "female"]`, `["from Europe", "from US", "from Asia"]`, `["uses Firefox", "uses Chrome", "uses Safari", "uses Internet Explorer"]`
>
> 整数编码为：
>
> `["male", "from US", "uses Internet Explorer"]` = [0 1 3]
>
> one-hot编码为：
>
> `["male", "from US", "uses Internet Explorer"]` = [0 0 0 1 0 1 0 0 0]

通过例2可以发现one-hot编码只是对每一个特征要求编码后序列中只有一个1。

### 评价

好处：



one hot编码是将类别变量转换为机器学习算法易于利用的一种形式的过程。

为什么使用one-hot编码来处理离散型特征?

为了解决上述的问题，使训练过程中不受到因为分类值表示的问题对模型产生的负面影响，引入独热码对分类型的特征进行独热码编码

使用one-hot编码的直接结果就是数据变稀疏



你可以要问，为什么编码不直接用数字来表示，example2里可以：{male, US} = [1 2]

整数编码，二进制编码

这有点像隐变量

### 代码

我们以example2为例

```python
from sklearn import preprocessing

# enc = preprocessing.OrdinalEncoder() # 整数编码
# enc = preprocessing.OneHotEncoder() # 独热编码

genders = ['female', 'male']
locations = ['from Africa', 'from Asia', 'from Europe', 'from US']
browsers = ['uses Chrome', 'uses Firefox', 'uses IE', 'uses Safari']

enc = preprocessing.OneHotEncoder(categories=[genders, locations, browsers])

integer_codes = enc.fit_transform([['female', 'from US', 'uses Safari'],
               ['male', 'from Europe', 'uses Safari']]).toarray()
print(integer_codes)
# [[1. 0. 0. 0. 0. 1. 0. 0. 0. 1.]
# [0. 1. 0. 0. 1. 0. 0. 0. 0. 1.]]

original_representation = enc.inverse_transform([[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]])
print(original_representation)
# [['female' 'from US' 'uses Safari']]
```

