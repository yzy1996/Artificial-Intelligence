import cma
import numpy as np
import matplotlib.pyplot as plt

# es = cma.CMAEvolutionStrategy(12 * [0], 0.5)

# while not es.stop():
#      solutions = es.ask()
#      es.tell(solutions, [cma.ff.rosen(x) for x in solutions])
#      es.logger.add()  # write data to disc to be plotted
#      es.disp()


# es.result_pretty()
# cma.plot()  # shortcut for es.logger.plot()
# plt.show()

es = cma.CMAEvolutionStrategy(3*[0], 0.5)

def F(x):
    return x[0] ** 2 + x[1] ** 2

while not es.stop():
     solutions = es.ask()
     for x in solutions:
          print(x)
     es.tell(solutions, [F(x) for x in solutions])
     es.logger.add()  # write data to disc to be plotted
     es.disp()
     break


es.result_pretty()
cma.plot()  # shortcut for es.logger.plot()
input()