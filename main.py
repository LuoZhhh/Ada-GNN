import argparse
import torch
import numpy as np
from model import prepare_data, SGC, SAGE, calc_f1
import time
from dataloader import ClusterDataloader
import learn2learn as l2l
from utils import save_best_model, load_best_model, softmax


def main(args):
    if args.gpu < 0:
        device = "cpu"
    else:
        device = "cuda:{}".format(args.gpu)

    graph, node_group = prepare_data(args)

    dataloader = ClusterDataloader(graph, node_group)

    in_feats = graph.ndata['feat'].shape[1]
    hid_feats = args.hid_feats
    out_feats = (graph.ndata['label'].max() + 1).item()

    loss_fcn = torch.nn.CrossEntropyLoss()

    if args.model == 'SGC':
        global_model = SGC(in_feats=in_feats, hid_feats=hid_feats, out_feats=out_feats)
    elif args.model == 'SAGE':
        global_model = SAGE(in_feats=in_feats, hid_feats=hid_feats, out_feats=out_feats)
    global_model = global_model.to(device)

    maml = l2l.algorithms.MAML(global_model, lr=args.lr_adapt, first_order=True)
    
    global_opt = torch.optim.Adam(maml.parameters(), lr=args.lr_global, weight_decay=args.weight_decay)

    best_epoch = 0
    best_val = 0

    start_time = time.time()

    graph = graph.int().to(device)

    for epoch in range(1, args.n_epochs + 1):
        meta_loss = 0
        sub_mics = []
        std_loss = 0
        local_losses = []

        for i, data in enumerate(dataloader):
            sub_train_nid, sub_val_nid, _ = data
            local_model = maml.clone()
            for step in range(args.num_steps):
                logits = local_model(graph, graph.ndata['feat']) # only compute the train set
                loss = loss_fcn(logits[sub_train_nid], graph.ndata['label'][sub_train_nid])
                local_model.adapt(loss)
            logits = local_model(graph, graph.ndata['feat']) # only compute the train set
            local_loss = loss_fcn(logits[sub_train_nid], graph.ndata['label'][sub_train_nid])

            pred = logits[sub_val_nid]

            if i == 0:
                overall_logits = pred.detach().cpu().numpy()
                overall_labels = graph.ndata['label'][sub_val_nid].detach().cpu().numpy()
            else:
                overall_logits = np.concatenate((overall_logits, pred.detach().cpu().numpy()), axis=0)
                overall_labels = np.concatenate((overall_labels, graph.ndata['label'][sub_val_nid].detach().cpu().numpy()), axis=0)

            sub_mic, _ = calc_f1(pred.detach().cpu().numpy(), graph.ndata['label'][sub_val_nid].detach().cpu().numpy())
            sub_mics.append(sub_mic)

            local_losses.append(local_loss)

        sub_mics = np.array(sub_mics)
        local_losses = np.array(local_losses)
        if args.fairness:
            weights = softmax(((1/sub_mics) / (1/sub_mics).sum())/args.tau)
        else:
            weights = np.ones(args.num_parts)
        meta_loss = (weights * local_losses).sum()

        global_loss = meta_loss
        global_opt.zero_grad()
        global_loss.backward()
        global_opt.step()

        used_time = time.time() - start_time
        memory = torch.cuda.max_memory_allocated()/1024/1024 if torch.cuda.is_available() else 0

        print(">> validation")
        val_mic, val_mac = calc_f1(overall_logits, overall_labels)

        print(">> epoch {}, loss {:.8f}, val_mic_f1 {:.6f}, val_mac_f1 {:.6f}, time {:.1f}s, memory {:.1f}MB".format(epoch, meta_loss.item(), val_mic, val_mac, used_time, memory))

        if val_mic > best_val:
            best_val = val_mic
            best_epoch = epoch
            save_best_model(global_model, path='saved_model/ada-{}.pt'.format(args.model))

        if epoch > best_epoch + 10:
            print(">> Converged at epoch {:d}.".format(epoch))
            break
    
    print(">> test")
    global_model = load_best_model(global_model, path='saved_model/ada-{}.pt'.format(args.model))
    maml = l2l.algorithms.MAML(global_model, lr=args.lr_adapt, first_order=True)
    local_mics = []
    
    for i, data in enumerate(dataloader):

        sub_train_nid, _, sub_test_nid = data
        local_model = maml.clone()
        for step in range(args.num_steps):
            logits = local_model(graph, graph.ndata['feat']) # only compute the train set
            loss = loss_fcn(logits[sub_train_nid], graph.ndata['label'][sub_train_nid])
            local_model.adapt(loss)
        logits = local_model(graph, graph.ndata['feat']) # only compute the train set
        local_loss = loss_fcn(logits[sub_train_nid], graph.ndata['label'][sub_train_nid])

        pred = logits[sub_test_nid]

        local_mic, _ = calc_f1(pred.detach().cpu().numpy(), graph.ndata['label'][sub_test_nid].detach().cpu().numpy())

        local_mics.append(local_mic)
        print(">> Subgroup {:d}'s local performance:{:.4f}".format(i, local_mic))

        if i == 0:
            overall_logits = pred.detach().cpu().numpy()
            overall_labels = graph.ndata['label'][sub_test_nid].detach().cpu().numpy()
        else:
            overall_logits = np.concatenate((overall_logits, pred.detach().cpu().numpy()), axis=0)
            overall_labels = np.concatenate((overall_labels, graph.ndata['label'][sub_test_nid].detach().cpu().numpy()), axis=0)
    
    test_mic, test_mac = calc_f1(overall_logits, overall_labels)
    print(">> test_mic_f1 {:.6f}, test_mac_f1 {:.6f}".format(test_mic, test_mac))
    








if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ada-GNN')
    parser.add_argument("--gpu", type=int, default=0,
            help="gpu")
    parser.add_argument("--dataset", type=str, default='arxiv', choices=["amazon", "arxiv"])
    parser.add_argument("--lr-global", type=float, default=0.005,
            help="global learning rate")
    parser.add_argument("--lr-adapt", type=float, default=0.005,
            help="adaption learning rate")
    parser.add_argument("--fairness", action='store_true', default=True,
            help="use ada-gnn-fair")
    parser.add_argument("--tau", type=float, default=0.005,
            help="temperature hyperparameter for fairness modulation")
    parser.add_argument("--model", type=str, default='SAGE',
            help="SGC or GraphSAGE")
    parser.add_argument("--n-epochs", type=int, default=1000,
            help="number of training epochs")
    parser.add_argument("--num-parts", type=int, default=3,
            help="partition size")
    parser.add_argument("--num-steps", type=int, default=5,
            help="local adaption steps")
    parser.add_argument("--hid-feats", type=int, default=128,
            help="hidden layer dimension")
    parser.add_argument("--weight-decay", type=float, default=0,
            help="Weight for L2 loss")
    args = parser.parse_args()
    print(args)

    main(args)