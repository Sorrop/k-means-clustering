from kmeans import kmeans
import matplotlib.pyplot as pl
import numpy as np


# put some random samples with different distributions in the plane
# in order to visualize as 3 groups
r1 = np.ndarray(shape=(200, 2))
r2 = np.ndarray(shape=(200, 2))
r3 = np.ndarray(shape=(200, 2))

r1x = 0.7 * np.random.randn(200) + 2
r1y = 0.5 * np.random.randn(200) + 4
r2x = 0.7 * np.random.randn(200) + 2
r2y = 0.5 * np.random.randn(200) + 2
r3x = 0.7 * np.random.randn(200) + 8
r3y = 0.5 * np.random.randn(200) + 6

for i in range(200):
    r1[i] = np.array([r1x[i], r1y[i]])
    r2[i] = np.array([r2x[i], r2y[i]])
    r3[i] = np.array([r3x[i], r3y[i]])

R = np.concatenate((r1, r2, r3), 0)

# plot them
BEFORE = pl.figure(1)
pl.plot(R[:, 0], R[:, 1], 'o')
BEFORE.show()

# apply kmeans clustering
g = kmeans(R, 3, 0.5)
g.compute()

# and plot the clusters in different colors
AFTER = pl.figure(2)
x = [item[0] for item in g.clusters[0]]
y = [item[1] for item in g.clusters[0]]
pl.plot(x, y, 'co')
x = [item[0] for item in g.clusters[1]]
y = [item[1] for item in g.clusters[1]]
pl.plot(x, y, 'yo')
x = [item[0] for item in g.clusters[2]]
y = [item[1] for item in g.clusters[2]]
pl.plot(x, y, 'mo')
pl.show()
