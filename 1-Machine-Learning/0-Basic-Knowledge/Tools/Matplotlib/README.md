## matplotlib库的使用



今天介绍一个mpl_toolkits库的使用，它可以用来绘制三维视图



最基本的使用方法如下：

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.show()
```



fig.add_subplot(111, projection='3d') 



plt.gca(*projection*='3d')



效果是一模一样的，都是创建了一个三维实例



plt.figure()的说明

可写可不写，不写在使用的时候会自动为你创建

写可以在括号内为图命名，默认是figure1





## 颜色

cmp 



