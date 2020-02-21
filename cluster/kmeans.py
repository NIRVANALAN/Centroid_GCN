import torch
import numpy as np
from kmeans_pytorch import kmeans

# data
data_size, dims, num_clusters = 1000, 2, 3
if torch.cuda.is_available():
    device = torch.device('cuda')
    x = torch.randn(data_size, dims).to(device)
# kmeans
cluster_ids_x, cluster_centers = kmeans(
    X=x, num_clusters=num_clusters, distance='cosine', device=device
)