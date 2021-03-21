'''Use of cma-es package pycma
'''

import cma
import numpy as np

# 初始化，均值和方差，也定义了维度
es = cma.CMAEvolutionStrategy(2 * [0], 0.3)

# 优化 Rosenbrock function（香蕉函数），能自适应维度
# es.optimize(cma.ff.rosen)

# 修改自定义目标函数
def F(x):
    return (x[0] - 1) ** 2 + (x[1]-2) ** 2

es.optimize(F)

# 打印结果
es.result_pretty()

# # 或者也可以简化为如下代码
# xopt, es = cma.fmin2(cma.ff.rosen, 3 * [0], 0.5)