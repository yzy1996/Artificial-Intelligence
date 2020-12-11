import re


# 第一步读取.MD文件



# 识别 $ $ 字符

# ![](https://latex.codecogs.com/svg.latex?%5Cinline%20G + xxx)

str1 = '![](https://latex.codecogs.com/svg.latex?%5Cinline%20G'

# 识别 $$ $$ 字符
# ![](https://latex.codecogs.com/svg.latex? + xxx)


str2 = '![](https://latex.codecogs.com/svg.latex?'

