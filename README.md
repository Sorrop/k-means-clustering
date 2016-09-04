# k-means-clustering


<h3><b>Implementation of classic K-means clustering algorithm in Python</b></h3>

This is the classic <a href="https://en.wikipedia.org/wiki/K-means_clustering">K-means clustering</a> method commonly utilized in unsupervised learning settings. We have a dataset of n samples with k features each and the objective is to group samples with their "closest" ones.

The algorithm works in succesive steps, starting initially with a set of random points that represent each cluster. Then each sample is assigned to a cluster according to its (euclidean) distance from the representative point. The next step is to find the centroid of each cluster and perform the previous step until a condition of convergence is achieved. At each step of centroid re-selection we attempt to minimize a utility function, namely the intra-cluster distances between points.

The algorithm is implemented in kmeans.py source by the <code>kmeans</code> class with input parameters <code>data_set</code> (of type <code>ndarray</code>), <code>n_clusters</code> (of integer type) and <code>limit</code> (a float). The method <code>compute()</code> performs the clustering and makes use of the methods <code>assign_to_clusters()</code>, <code>calc_utility_function()</code> and <code>calc_new_centroids()</code> in a serial fashion until succesive optimizations of the utility function reach a difference below a user-defined <code>limit</code>. 

Because of the random starting points it is possible to reach a local optimum that doesn't correspond to the global optimal clustering. To address this in <code>experiment.py</code> we run the algorithm 5 times on the input dataset and choose the resulting clustering that corresponds to the lowest utility function value.

Packages needed to run <code>kmeans.py</code> and <code>experiment.py</code>: <a href="http://www.numpy.org/">numpy</a>, <a href="http://matplotlib.org/">matplotlib</a>

