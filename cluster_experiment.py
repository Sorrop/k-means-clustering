from kmeans import Kmeans
import matplotlib.pyplot as pl
import numpy as np
import timeit
import math


# put some random samples with different distributions in the plane
# in order to visualize as 3 groups
r1 = np.ndarray(shape=(200, 2))
r2 = np.ndarray(shape=(200, 2))
r3 = np.ndarray(shape=(200, 2))

r1x = 0.7 * np.random.randn(200) + 2
r1y = 0.5 * np.random.randn(200) + 4
r2x = 0.7 * np.random.randn(200) + 4
r2y = 0.5 * np.random.randn(200) + 2
r3x = 0.7 * np.random.randn(200) + 5
r3y = 0.5 * np.random.randn(200) + 5

for i in range(200):
    r1[i] = np.array([r1x[i], r1y[i]])
    r2[i] = np.array([r2x[i], r2y[i]])
    r3[i] = np.array([r3x[i], r3y[i]])

R = np.concatenate((r1, r2, r3), 0)

# plot them
BEFORE = pl.figure(1)
pl.plot(R[:, 0], R[:, 1], 'o')
BEFORE.show()

# apply kmeans clustering 5 times to eliminate local optimum convergence
objective_val = math.inf
times = []
for i in range(5):
    start = timeit.default_timer()
    g = Kmeans(R, 3, 0.5)
    g.compute()
    times.append(timeit.default_timer() - start)

    if g.objective_hist[-1] < objective_val:
        objective_val = g.objective_hist[-1]
        clusters = g.clusters

time_of_exec = sum(times) / len(times)
print('Kmeans.compute() in %f seconds, on the average' % (time_of_exec))


# and plot the clusters in different colors
AFTER = pl.figure(2)
x = [item[0] for item in clusters[0]]
y = [item[1] for item in clusters[0]]
pl.plot(x, y, 'co')
x = [item[0] for item in clusters[1]]
y = [item[1] for item in clusters[1]]
pl.plot(x, y, 'yo')
x = [item[0] for item in clusters[2]]
y = [item[1] for item in clusters[2]]
pl.plot(x, y, 'mo')
pl.show()
