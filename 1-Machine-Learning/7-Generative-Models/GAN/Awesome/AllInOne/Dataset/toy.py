# 8 Gaussians

import numpy as np
import matplotlib.pyplot as plt


def toy_dataset(DATASET='8gaussians', size=256):

    if DATASET == '8gaussians':

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

        centers = [(2 * x, 2 * y) for x, y in centers]
        dataset = []

        for i in range(size):
            point = np.random.randn(2)*.02
            center = np.random.default_rng().choice(centers)
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        dataset /= 1.414  # stdev

    if DATASET == '25gaussians':

        dataset = []
        for i in range(size):
            for x in range(-2, 3):
                for y in range(-2, 3):
                    point = np.random.randn(2)*0.05
                    point[0] += 2 * x
                    point[1] += 2 * y
                    dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        np.random.shuffle(dataset)
        dataset /= 2.828  # stdev

    return dataset


if __name__ == '__main__':

    dataset = toy_dataset(DATASET='8gaussians', size=512)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(dataset[:, 0], dataset[:, 1], 'ko', alpha=0.1)
    # plt.xlim([-2, 2])
    # plt.ylim([-2, 2])
    plt.axis('off')
    ax.set_aspect(1)
    plt.show()
