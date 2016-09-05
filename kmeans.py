import numpy as np
from scipy.spatial import distance
import math


class Kmeans():
    '''
    Implementation of classical k-means clustering algorithm
    parameters : dataset n x m ndarray of n samples and m features
    n_clusters : number of clusters to assign samples to
    limit : tolerance between successive iterations
    '''
    def __init__(self, dataset, n_clusters, limit):

        self.dataset = dataset
        self.n_clusters = n_clusters
        self.limit = limit

        # dictionary to hold each cluster as a list of samples
        self.clusters = {i: [] for i in range(self.n_clusters)}

        # the centroids of each cluster
        self.centroids = np.ndarray((n_clusters, dataset.shape[1]))

        # successive values of objective function. increases in size  by 1
        # in each iteration. Unclusterised data corresponds
        # to a value of Inf for objective function
        self.objective_hist = [math.inf]

    def assign_to_clusters(self):
        '''
        Assign each sample in the data set to a cluster according
        to distance from the cluster's centroid
        '''

        # vectorized computation of each sample's distance to centroids
        # using scipy's cdist function
        distances_to_centroids = distance.cdist(
            self.dataset, self.centroids, metric='sqeuclidean')

        # assignment of each sample to appropriate cluster
        for i in range(distances_to_centroids.shape[0]):
            appropriate_cluster = np.argmin(distances_to_centroids[i])
            self.clusters[appropriate_cluster].append(self.dataset[i])

    def calc_objective_function(self):
        '''
        Compute the sum of distances of each sample
        to the centroid of each cluster. The goal is to minimize it.
        '''

        total_sum = 0
        for cluster, samples in self.clusters.items():
            centroid = self.centroids[cluster]

            for i in range(len(samples)):
                total_sum += np.linalg.norm(samples[i] - centroid)

        return total_sum

    def calc_new_centroids(self):
        '''
        Obtain new centroids from existing clusters
        '''

        # we calculate new centroids by obtaining the centers
        # of each (each) cluster
        centers = np.ndarray(shape=self.centroids.shape)

        for key, samples in self.clusters.items():

            temp_sam = np.array(samples)
            # that is the mean of each feature
            temp_mean = np.mean(temp_sam, axis=0)
            centers[key] = np.array(temp_mean)

        # the new centroid is the sample in the cluster that is closest
        # to the mean point
        for i in range(centers.shape[0]):

            distances = [np.linalg.norm(centers[i] - sample)
                         for sample in self.clusters[i]]
            new_centroid = distances.index(min(distances))
            self.centroids[i] = self.clusters[i][new_centroid]

        # clusters dictionary must empty in order to repopulate
        self.clusters = {i: [] for i in range(self.n_clusters)}

    def compute(self):
        '''
        Core method that computes the clusters of the dataset
        '''

        # initialize centroids by randomly choosing #n_clusters samples
        # from dataset
        self.centroids = self.dataset[np.random.choice(self.dataset.shape[0],
                                                       size=self.n_clusters,
                                                       replace=False), :]
        # perform one step
        self.assign_to_clusters()
        self.objective_hist.append(self.calc_objective_function())

        # at this point objective_hist = [Inf, x<Inf] so
        # there is no convergence
        converged = False
        while not converged:
            self.calc_new_centroids()
            self.assign_to_clusters()
            self.objective_hist.append(self.calc_objective_function())
            converged = math.isclose(self.objective_hist[-1],
                                     self.objective_hist[-2],
                                     abs_tol=self.limit)
