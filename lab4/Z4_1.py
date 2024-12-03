import matplotlib.pyplot as plt
from sklearn.metrics import DistanceMetric
import numpy as np
from sklearn.datasets import make_classification
from mpi4py import MPI


def custom_sequential_kmeans(X, n_clusters, max_iter=100, tol=1e-4, random_state=42):
    dist_metric = DistanceMetric.get_metric("euclidean")
    
    initial_indices = np.random.choice(len(X), n_clusters, replace=False)
    centroids = X[initial_indices]
    prev_centroids = np.zeros_like(centroids)

    for iteration in range(max_iter):
        distances = dist_metric.pairwise(X, centroids)
        cluster_labels = np.argmin(distances, axis=1)

    for i in range(n_clusters):
        cluster_points = X[cluster_labels == i]
        if len(cluster_points) > 0:
            centroids[i] = np.mean(cluster_points, axis=0)

        if np.all(np.linalg.norm(centroids - prev_centroids, axis=1) < tol):
            print(f"Converged after {iteration+1} iterations.")
            break

    prev_centroids = centroids.copy()

    return cluster_labels, centroids


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    dist_metric = DistanceMetric.get_metric("euclidean")
    
    t1 = MPI.Wtime()

    if rank == 0:
        X, y = make_classification(n_samples=10000, n_features=2, n_informative=2, n_redundant=0, random_state=1410)
        plt.scatter(X[:, 0],X[:, 1], c="blue")
        # subsets
        ave, res = divmod(X.shape[0], size)
        counts = [ave + 1 if p < res else ave for p in range(size)]
        # print(counts)
        starts = [sum(counts[:p]) for p in range(size)]
        ends = [sum(counts[:p+1]) for p in range(size)]
        X = [X[starts[p]:ends[p]] for p in range(size)]
        
        cluster_labels, centroids = custom_sequential_kmeans(X, n_clusters=n_clusters, random_state=1410)

        local_data = comm.scatter(X, root=0)
        centroids = comm.bcast(centroids, root=0)
        comm.Barrier()


        lab = comm.gather(cluster_labels)
        cent = comm.gather(centroids)
        comm.Barrier()

        if rank==0:
            lab = np.asarray(lab).ravel().reshape(num_row(X),-1) 
            cent = np.asarray(cent, dtype='i').ravel().reshape(num_row(X),-1) 
            new_cl, new_cent = custom_sequential_kmeans(X, n_clusters=n_clusters, random_state=1410)
            check = np.array_equal(new_centroids, centroids) 
        
        else:
            check = None
            new_centroids = None
        
        check = comm.bcast(check, root=0)
        comm.Barrier()
        
        if check == False:
            centroids = new_centroids
            time+=1
        else:
            t2 = MPI.Wtime()
            if rank ==0:
                print("Run time: {}".format(t2-t1))

        
    else:
        X = None
        centroids = None
        membership = None

    # X, _ = make_classification(n_samples=10000, n_features=2, n_informative=2, n_redundant=0, random_state=1410)
    # n_clusters = 3
    # cluster_labels, centroids = custom_sequential_kmeans(X, n_clusters=n_clusters, random_state=1410)

    # plt.scatter(X[:,0], X[:,1], c=cluster_labels, cmap='viridis')
    # plt.scatter(centroids[:,0], centroids[:,1], c='r', s=80)
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()