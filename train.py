from cluster import *  # import cluster
import argparse
import pathlib
from pathlib import Path
import time
import pdb
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args, load_data

from models import create_model
from models.utils import EarlyStopping


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits, _ = model(features)
        # logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def main(args):
    # load and preprocess dataset
    data = load_data(args)
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    np.save(f'{args.dataset}_labels', labels)
    if hasattr(torch, 'BoolTensor'):
        train_mask = torch.BoolTensor(data.train_mask)
        val_mask = torch.BoolTensor(data.val_mask)
        test_mask = torch.BoolTensor(data.test_mask)
    else:
        train_mask = torch.ByteTensor(data.train_mask)
        val_mask = torch.ByteTensor(data.val_mask)
        test_mask = torch.ByteTensor(data.test_mask)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    cluster_interval = args.cluster_interval
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
              train_mask.int().sum().item(),
              val_mask.int().sum().item(),
              test_mask.int().sum().item()))

    if args.early_stop:
        stopper = EarlyStopping(patience=100)
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        device = features.device
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()

    # graph preprocess and calculate normalization factor
    g = data.graph
    # add self loop
    if not args.no_self_loop:
        print('add self-loop')
        g.remove_edges_from(nx.selfloop_edges(g))
        g.add_edges_from(zip(g.nodes(), g.nodes()))
    g = DGLGraph(g)
    n_edges = g.number_of_edges()
    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    # # create GCN model
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    model = create_model(args.arch, g,
                         num_layers=args.num_layers,
                         in_dim=in_feats,
                         num_hidden=args.num_hidden,
                         num_classes=n_classes,
                         heads=heads,
                         activation=F.elu,
                         feat_drop=args.in_drop,
                         attn_drop=args.attn_drop,
                         negative_slope=args.negative_slope,
                         residual=args.residual)

    print(model)
    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # Step 1. initilization with GCN
    # init graph feat
    dur = []
    centroid_emb, hidden_emb, cluster_ids = [], [], []
    att = []
    for epoch in range(args.epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # cluster
        # forward
        if epoch < args.init_feat_epoch:
            # logits = model(features)
            logits, hidden_h = model(features)
        else:
            if epoch == args.init_feat_epoch or epoch % cluster_interval == 0:
                cluster_ids_x, cluster_centers = cluster(
                    X=hidden_h.detach(), num_clusters=args.cluster_number, distance='cosine', method=args.cluster_method)  # TODO: fix zero norm embedding
                centroid_emb.append(cluster_centers.detach().cpu().numpy())
                hidden_emb.append(hidden_h.detach().cpu().numpy())
                cluster_ids.append(cluster_ids_x.detach().cpu().numpy())
                pass
            logits, hidden_h = model(
                features, cluster_ids_x, cluster_centers, att)
            # logits, hidden_h = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        # loss.backward(retain_graph=True)
        loss.backward(retain_graph=False)
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)
        if args.fastmode:
            val_acc = accuracy(logits[val_mask], labels[val_mask])
        else:
            val_acc = evaluate(model, features, labels, val_mask)
            if args.early_stop:
                if stopper.step(val_acc, model):
                    break
        # acc = evaluate(model, features, labels, val_mask)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                             val_acc, n_edges / np.mean(dur) / 1000))

    print()
    acc = evaluate(model, features, labels, test_mask)
    print("Test accuracy {:.2%}".format(acc))
    prefix = 'embedding'
    np.save(Path(prefix, f'{args.dataset}_centroid_emb'),
            np.array(centroid_emb))
    np.save(Path(prefix, f'{args.dataset}_hidden_emb'), np.array(hidden_emb))
    np.save(Path(prefix, f'{args.dataset}_att'), np.array(att))
    np.save(Path(prefix, f'{args.dataset}_cluster_ids'), np.array(cluster_ids))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    # parser.add_argument("--dropout", type=float, default=0.5,
    #                     help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--no-self-loop", action='store_true',  # !MUST IN GAT
                        help="graph self-loop (default=False)")
    # cluster
    parser.add_argument("--cluster_method", type=str, default='kmeans',
                        help="Cluster method, default=kmeans")
    parser.add_argument("--cluster-interval", type=int, default=25,
                        help="interval of calculating cluster centroid")
    parser.add_argument("--cluster-number", type=int, default=6,
                        help="interval of calculating cluster centroid")

    # attention
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=8,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--fastmode', action="store_true", default=False,
                        help="skip re-evaluate the validation set")

    parser.add_argument("--init-feat-epoch", type=int, default=25,
                        help="stage 1 training epoch number")
    # MODEL
    parser.add_argument("--arch", type=str, default='gcn',
                        help='arch of gcn model, default: gcn')
    # parser.add_argument("--num-classes", type=int, default=1500,
    #                     help="Number of clusters, for Reddit 1500 by default")
    # parser.add_argument("--batch_size", type=int, default=5000,
    #                     help="Batch size")
    args = parser.parse_args()
    print(args)

    main(args)
