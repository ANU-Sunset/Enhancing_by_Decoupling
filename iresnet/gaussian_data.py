import torch
import sklearn
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

"""
def generateRandomPointFromAnnulus():
    data = datasets.make_circles(n_samples=10000, factor=.5, noise=0.08)[0]
    data = data.astype("float32")
    data *= 3
    data = torch.from_numpy(data)
    return data


def eight_gaussian():
    rng = np.random.RandomState()
    scale = 4.
    centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
               (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                     1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
    centers = [(scale * x, scale * y) for x, y in centers]
    for i in range(len(centers)):
        for k in range(2 - 2):
            centers[i] = centers[i] + (0,)

    dataset = []
    for i in range(10000):
        point = rng.randn(2) * 0.5
        idx = rng.randint(8)
        center = centers[idx]
        point[0] += center[0]
        point[1] += center[1]
        dataset.append(point)
    dataset = np.array(dataset, dtype="float32")
    dataset /= 1.414
    dataset = torch.from_numpy(dataset)
    return dataset

"""

import torch
import random
import matplotlib.pyplot as plt
import numpy as np


def generateRandomPointFromAnnulus():
    result = torch.zeros(100000, 2)
    count = 0
    while True:
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x * x + y * y <= 1 * 1 and x * x + y * y >= 0.9 * 0.9 or \
                x * x + y * y <= 0.7 * 0.7 and x * x + y * y >= 0.6 * 0.6 or \
                x * x + y * y <= 0.4 * 0.4 and x * x + y * y >= 0.3 * 0.3:
            result[count, 0] = x
            result[count, 1] = y
            count += 1
        if count == 100000:
            return result


def eight_gaussian():
    result = torch.zeros(80000, 2)
    cov = [[1, 0], [0, 1]]
    for i in range(10000):
        result[i] = torch.from_numpy(np.random.multivariate_normal(mean=[10, 0], cov=cov))
        result[i + 10000] = torch.from_numpy(
            np.random.multivariate_normal(mean=[5 * np.sqrt(2), 5 * np.sqrt(2)], cov=cov))
        result[i + 20000] = torch.from_numpy(np.random.multivariate_normal(mean=[0, 10], cov=cov))
        result[i + 30000] = torch.from_numpy(
            np.random.multivariate_normal(mean=[-5 * np.sqrt(2), 5 * np.sqrt(2)], cov=cov))
        result[i + 40000] = torch.from_numpy(np.random.multivariate_normal(mean=[-10, 0], cov=cov))
        result[i + 50000] = torch.from_numpy(
            np.random.multivariate_normal(mean=[-5 * np.sqrt(2), -5 * np.sqrt(2)], cov=cov))
        result[i + 60000] = torch.from_numpy(np.random.multivariate_normal(mean=[0, -10], cov=cov))
        result[i + 70000] = torch.from_numpy(
            np.random.multivariate_normal(mean=[5 * np.sqrt(2), -5 * np.sqrt(2)], cov=cov))
    return result


a = generateRandomPointFromAnnulus()
b = eight_gaussian()

torch.save(a, './rings.pt')
torch.save(b, './gaussian.pt')
ax = plt.subplot(121)
ax.scatter(a[:, 0], a[:, 1], s=1, color='#49b6ff')
ax = plt.subplot(122)
ax.scatter(b[:, 0], b[:, 1], s=1, color='#49b6ff')
plt.show()
