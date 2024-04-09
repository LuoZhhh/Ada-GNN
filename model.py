import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import pickle
from collections import Counter
import numpy as np
from dgl.nn.pytorch.conv import SAGEConv, SGConv
from sklearn.metrics import f1_score
from partition import metis_partition, get_node_group


def calc_f1(logits, labels):

    logits, labels = np.squeeze(logits), np.squeeze(labels)
    y_true = labels

    is_multi_label = len(labels.shape) > 1
    if is_multi_label:
        logits[logits > 0] = 1
        logits[logits <= 0] = 0
        y_pred = logits
    else:
        y_pred = np.argmax(logits, axis=-1)

    def _f1_score(average):
        return f1_score(y_true, y_pred, average=average)

    mic_f1, mac_f1 = map(_f1_score, ["micro", "macro"])

    return mic_f1, mac_f1



class SGC(nn.Module):

    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = SGConv(in_feats,
                hid_feats,
                k=2,
                cached=True,
                bias=True
            )
        self.conv2 = SGConv(hid_feats,
                out_feats,
                k=1,
                cached=False,
                bias=True
            )


    def forward(self, graph, x):
        h1 = self.conv1(graph, x)
        h2 = self.conv2(graph, h1)



        return h2


class SAGE(nn.Module):

    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean',
            norm=nn.LayerNorm(hid_feats, elementwise_affine=True),
            activation=F.relu
        )
        # self.conv2 = SAGEConv(
        #     in_feats=hid_feats, out_feats=hid_feats, aggregator_type='mean',
        #     norm=nn.LayerNorm(hid_feats, elementwise_affine=True),
        #     activation=F.relu
        # )

        self.conv3 = SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean'
        )


    def forward(self, graph, x):
        h = self.conv1(graph, x)
        # h = self.conv2(graph, h)
        h = self.conv3(graph, h)
        return h


def prepare_data(args):
    data_path = 'dataset/'
    with open(data_path+args.dataset+'/graph.pkl', 'rb') as f:
        g = pickle.load(f)

    print("########################")
    print(">> Load Graph...")
    train_nids = g.nodes()[g.ndata['train_mask']]
    val_nids = g.nodes()[g.ndata['val_mask']]
    test_nids = g.nodes()[g.ndata['test_mask']]
    print('>> train_nids:', len(train_nids))
    print('>> val_nids:', len(val_nids))
    print('>> test_nids:', len(test_nids))
    print(">> classes:", (g.ndata['label'].max() + 1).item())
    print("########################")

    # metis partition
    num_cluster, membership = metis_partition(g, args)

    node_group = get_node_group(num_cluster, membership)

    g = generate(g, node_group)

    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    return g, node_group


def evaluate(epoch, args, model, feats, labels, train, val, test):
    with torch.no_grad():
        batch_size = args.eval_batch_size
        if batch_size <= 0:
            pred = model(feats)
        else:
            pred = []
            num_nodes = labels.shape[0]
            n_batch = (num_nodes + batch_size - 1) // batch_size
            for i in range(n_batch):
                batch_start = i * batch_size
                batch_end = min((i + 1) * batch_size, num_nodes)
                batch_feats = [feat[batch_start: batch_end] for feat in feats]
                pred.append(model(batch_feats))
            pred = torch.cat(pred)
        
        mic, mac = calc_f1(pred[test].detach().cpu().numpy(), labels[test].detach().cpu().numpy())
        val_mic, _ = calc_f1(pred[val].detach().cpu().numpy(), labels[val].detach().cpu().numpy())
        return val_mic, mic, mac


def generate(graph, node_group):
    
    distributions = []

    num_label = graph.ndata['label'].max().item() + 1
    
    print(">> Original feature dimension: ", graph.ndata['feat'].shape[1])

    ############################## label distribution ##################################
    indicators = []
    for i, cluster_nids in enumerate(node_group):
        sub_graph = graph.subgraph(cluster_nids)
        labels = sub_graph.ndata['label'].numpy()

        distribution = np.array([Counter(labels)[key] if key in Counter(labels) else 0 for key in range(num_label)]) / sub_graph.num_nodes()
        distributions.append(distribution)
        # print(type(distribution[0]))
        indicators.append(list(map(int, list('{:08b}'.format(i)))))  # cluster indicator

    new_feat = torch.cat((graph.ndata['feat'], torch.zeros((graph.num_nodes(), num_label + 8))), dim=-1)

    for nodes, distribution in zip(node_group, distributions):
        new_feat[nodes, -int(num_label+8):-8] = torch.tensor(distribution, dtype=torch.float32)
    for nodes, indicator in zip(node_group, indicators):
        new_feat[nodes, -8:] = torch.tensor(indicator, dtype=torch.float32)

    graph.ndata['feat'] = new_feat

    print(">> Generated feature dimension: ", graph.ndata['feat'].shape[1])
    return graph
