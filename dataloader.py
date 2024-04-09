import numpy as np
import random
import torch



class ClusterDataloader:

    def __init__(self, graph, node_group):

        self.num_cluster = len(node_group)
        self.node_group = node_group

        self.train_mask = graph.ndata['train_mask']
        self.val_mask = graph.ndata['val_mask']
        self.test_mask = graph.ndata['test_mask']

        self.index = 0

    def __len__(self):
        return self.num_cluster

    def __iter__(self):
        np.random.seed(2275)

        self.index = 0
        return self

    def __next__(self):
        
        if self.index == self.num_cluster:
            raise StopIteration
        index = self.index
        self.index += 1

        sub_train_nid = torch.tensor(self.node_group[index])[(self.train_mask[self.node_group[index]] == True).nonzero()[:,0]].tolist()
        sub_val_nid = torch.tensor(self.node_group[index])[(self.val_mask[self.node_group[index]] == True).nonzero()[:,0]].tolist()
        sub_test_nid = torch.tensor(self.node_group[index])[(self.test_mask[self.node_group[index]] == True).nonzero()[:,0]].tolist()

        
        
        return (sub_train_nid, sub_val_nid, sub_test_nid)

