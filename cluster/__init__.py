from .kmeans import kmeans


cluster_methods = {
    'kmeans': kmeans
}


def cluster(X, num_clusters, device, distance='cosine', method='kmeans') -> tuple:
    """ Embedding Custer
    return cluster_id, cluster_centroid
    """
    assert method in cluster_methods
    cluster_ids_x, cluster_centers = cluster_methods[method](
        X, num_clusters, distance, device=device
    )
    return cluster_ids_x, cluster_centers

__all__ = ['cluster'] # ! from cluster import *