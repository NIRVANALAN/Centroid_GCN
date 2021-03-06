from .gcn import GCN
from .g_centroid import GATCentroid
from . import utils
from .gat import GAT

# TODO: add centroid-driven message passing graph net
FACTORY = {
    'gcn': GCN,
    'gat': GAT,
    'centroid': GATCentroid}


def create_model(name, g, **kwargs):
    if name not in FACTORY:
        raise NotImplementedError(f'{name} not in arch FACTORY')
    return FACTORY[name](g, **kwargs)
