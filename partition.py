import numpy as np
import dgl



def metis_partition(graph, args):

    num_parts = args.num_parts
    print(args.dataset)

    print(">> Metis Clustering...")
    
    dgl.distributed.partition.partition_graph(graph, 
                                args.dataset, 
                                num_parts,       
                                out_path="metis/output/", reshuffle=False,
                                balance_ntypes=graph.ndata['train_mask'],
                                balance_edges=True)
                                
    membership = np.load("metis/output/node_map.npy")

    num_cluster = num_parts
    print(">> Metis Clustering Finished.")

    return num_cluster, membership



def get_node_group(num_cluster, membership):

    node_group = [[] for _ in range(num_cluster)]
    for nid, cid in enumerate(membership):
        node_group[cid].append(nid)

    return node_group
