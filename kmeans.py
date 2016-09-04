import numpy as np


class kmeans():
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

        # values of utility function. increases in size  by 1
        # in each iteration
        self.util_func_vals = []

    def assign_to_clusters(self):

        for idx, sample in enumerate(self.dataset):

            distances = []
            # for each sample we compute its distance from every centroid
            for centroid in self.centroids:
                distances.append(np.linalg.norm(sample - centroid))

            # and assign it to the appropriate cluster
            appropriate_cluster = distances.index(min(distances))

            self.clusters[appropriate_cluster].append(sample)

    def calc_utility_function(self):

        total_sum = 0
        # utility function is the sum of intra-cluster distances
        # the goal is to minimize it
        for cluster, samples in self.clusters.items():

            for i in range(len(samples)):

                for j in range(i + 1, len(samples)):

                    total_sum += np.linalg.norm(samples[i] - samples[j])

        return total_sum

    def calc_new_centroids(self):
        # we calculate new centroids by obtaining the centers 
        # of each (each) cluster
        centers = np.ndarray(shape=self.centroids.shape)

        for key, samples in self.clusters.items():
            temp_mean = []
            temp_sam = np.array(samples)
            # that is the mean of each feature
            for i in range(self.dataset.shape[1]):
                temp_mean.append(sum(temp_sam[:, i]) / temp_sam.shape[0])
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
        # core method that computes the clusters
        # initialize centroids by randomly choosing #n_clusters samples
        # from dataset
        self.centroids = self.dataset[np.random.choice(self.dataset.shape[0],
                                                       size=self.n_clusters,
                                                       replace=False), :]
        # apply the first two steps of the algorithm
        self.assign_to_clusters()
        self.util_func_vals.append(self.calc_utility_function())

        self.calc_new_centroids()

        self.assign_to_clusters()
        self.util_func_vals.append(self.calc_utility_function())

        # and continue until the succesive value difference of utility function
        # becomes lower than the user specified limit
        while abs(self.util_func_vals[-1] - self.util_func_vals[-2]) > self.limit:

            self.calc_new_centroids()

            self.assign_to_clusters()

            self.util_func_vals.append(self.calc_utility_function())
