from __future__ import division
import sys
import numpy as np
from scipy.stats import multivariate_normal

def standardize(x):
    return (x - x.min(axis=0))/x.ptp(axis=0)
    
def random_centroids(x,k):
    np.random.shuffle(x)
    return x[0:k,:]
    
def cluster_assignment(x,c):
    return ((x[:,np.newaxis] - c)**2).sum(axis=2).argmin(axis=1)
    
def updated_centroids(x,clusterIndex,k,d):
    c= np.zeros((k,d))
    for i in range(k):
        c[i] = x[clusterIndex == i].mean(axis=0)
    return c

def kmeansClustering(x,centroids,k, d, maxIter):
    for b in range(maxIter):
        # finding the closet centroid
        clusterIndex = cluster_assignment(x,centroids)
                
        # Update centroid based on cluster assignment
        centroids = updated_centroids(x,clusterIndex,k,d)
        # writing the initial centroid in the file
        np.savetxt('centroids-' + str(b+1) + '.csv', centroids, delimiter=',')



if __name__ == '__main__':
    # load data
    x = np.genfromtxt(sys.argv[1], delimiter=',')
    k = 5 # as mentioned in the problem
    
    # number of rows and columns in the data set
    n, d = x.shape
    maxIter = 10
    # first step is to randomly initialise the k centroids
    # we have K centeroids
    centroids = random_centroids(x,k)
    
    # Initializing parameters for EM algo
    mu = centroids
    pi = np.ones(k) * 1.0/k
    sigma = np.array([np.identity(d) for i in range(k)])
    
    # applying k means clustering
    
    kmeansClustering(x,centroids,k, d, maxIter)
    em_gmm_clustering(x, pi, mu, sigma, k, n, d, maxIter)