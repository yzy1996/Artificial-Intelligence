## 经常遇到的错误

> RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!

首先有几个很容易犯错的地方，输入数据和模型是否放到GPU上，

```python
data = torch.ones(10, 10, device=device)
model = Model().to(device)

# 这两点要写对
```



另外核心解决思路就是找到那个异常的位置

虽然我们可以通过对每个变量 print(xx.device)，这显然效率很低



```javascript
!pip install torchsnooper

import torch
import torchsnooper

@torchsnooper.snoop()
def myfunc(mask, x):
    y = torch.zeros(6)
    y.masked_scatter_(mask, x)
    return y
```

