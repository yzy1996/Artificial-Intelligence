import cma

xopt, es = cma.fmin2(cma.ff.rosen, 8 * [0], 0.5)