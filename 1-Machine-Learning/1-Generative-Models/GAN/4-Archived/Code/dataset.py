
import numpy as np

centers = [
    (1., 0),
    (-1., 0),
    (0, 1.),
    (0, -1.),
    (1./np.sqrt(2), 1./np.sqrt(2)),
    (1./np.sqrt(2), -1./np.sqrt(2)),
    (-1./np.sqrt(2), 1./np.sqrt(2)),
    (-1./np.sqrt(2), -1./np.sqrt(2))
]


print(np.random.default_rng().choice(centers))