'''Use of cma-es package pycma
'''

import cma

es = cma.CMAEvolutionStrategy(8 * [0], 0.5)
es.optimize(cma.ff.rosen)

es.result_pretty()