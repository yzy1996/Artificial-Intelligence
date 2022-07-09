# Pytorch

推荐学习材料：https://github.com/lyhue1991/eat_pytorch_in_20_days



## 1.查看版本

```bash
import torch
print(torch.__version__)
```



推荐一个插件 [einops](https://github.com/arogozhnikov/einops)

## 2. 基本函数

torch.tensor 具有属性

```python
# 是否可以求导
requires_grad=True/False 

# 操作名称
grad_fn = <AddBackward0> / <MulBackward0> / <MeanBackward0> / <SumBackward0>
```

计算导数使用

```python
# 计算
.backward()

# 结果
.grad
```



判断两个 tensor 相等



torch.equal(x1, x2)



有三种构建模型的方法：



- 使用nn.Sequential按层顺序构建模型，

- 继承nn.Module基类构建自定义模型，

- 继承nn.Module基类构建模型并辅助应用模型容器(nn.Sequential,nn.ModuleList,nn.ModuleDict)进行封装。



> nn.Sequential

```python
def create_net():
    net = nn.Sequential()
    net.add_module("linear1",nn.Linear(15,20))
    net.add_module("relu1",nn.ReLU())
    net.add_module("linear2",nn.Linear(20,15))
    net.add_module("relu2",nn.ReLU())
    net.add_module("linear3",nn.Linear(15,1))
    net.add_module("sigmoid",nn.Sigmoid())
    return net

net = create_net()

from torchkeras import summary
summary(net,input_shape=(15,))
```



> nn.Module

```python
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3)
        self.pool = nn.MaxPool2d(kernel_size = 2,stride = 2)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5)
        self.dropout = nn.Dropout2d(p = 0.1)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1,1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64,32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32,1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        y = self.sigmoid(x)
        return y
        
net = Net()
```



> Train

```

```





## 3. 搭建Le-Net





custom Module subclass





## 数据处理

**(1) numpy to tensor**



```python
torch.from_numpy()
# 会共享相同的内存，修改tensor也会对应修改ndarray
# NumPy ndarray

>>> a = numpy.array([1, 2, 3])
>>> t = torch.from_numpy(a)
>>> t
tensor([ 1,  2,  3])
>>> t[0] = -1
>>> a
array([-1,  2,  3])
```

```python
torch.as_tensor()
# Can be a list, tuple, NumPy ndarray, scalar, and other types.

>>> a = numpy.array([1, 2, 3])
>>> t = torch.as_tensor(a) # 不复制
>>> t
tensor([ 1,  2,  3])
>>> t[0] = -1
>>> a
array([-1,  2,  3])

>>> a = numpy.array([1, 2, 3])
>>> t = torch.as_tensor(a, device=torch.device('cuda')) # 不复制
>>> t
tensor([ 1,  2,  3])
>>> t[0] = -1
>>> a
array([1,  2,  3])
```



```python
torch.Tensor()
# copy data

>>> a = numpy.array([1, 2, 3])
>>> t = torch.Tensor(a)
>>> t
tensor([ 1,  2,  3])
>>> t[0] = -1
>>>
array([1,  2,  3])
```





model.named_parameters()

会输出 weight and bias



net.parameters()



model.state_dict() 带括号和不带括号

model.parameters()



_latents.z.weight





for var_name in optimizer.state_dict():
    print(var_name,'\t',optimizer.state_dict()[var_name])

，不同的是，model.parameters()方法返回的是一个生成器generator，每一个元素是从开头到结尾的参数，parameters没有对应的key名称，是一个由纯参数组成的generator，而state_dict是一个字典，包含了一个key。



for para in model.named_parameters(): # 返回的每一个元素是一个元组 tuple 
    '''
    是一个元组 tuple ,元组的第一个元素是参数所对应的名称，第二个元素就是对应的参数值



named_parameters，而且parameters正是通过named_parameters来实现的，





https://blog.csdn.net/qq_27825451/article/details/95888267
