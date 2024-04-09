import argparse
import numpy as np
import pickle
import torch
import torch.nn as nn
from dataloader import ClusterDataloader
from model import prepare_data, SGC, SAGE, calc_f1
from sklearn.metrics import f1_score
from utils import save_best_model, load_best_model



def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)[mask] # only compute the evaluation set
        labels = labels[mask]

        logits, labels = np.squeeze(logits.cpu()), np.squeeze(labels.cpu())
        
        y_true = labels
        is_multi_label = len(labels.shape) > 1
        if is_multi_label:
            logits[logits > 0] = 1
            logits[logits <= 0] = 0
            y_pred = logits
        else:
            y_pred = np.argmax(logits, axis=1)

        def _f1_score(average):
            return f1_score(y_true, y_pred, average=average)
        
        mic_f1, mac_f1 = map(_f1_score, ["micro", "macro"])

        return mic_f1, mac_f1


def main(args):
    if args.gpu < 0:
        device = "cpu"
    else:
        device = "cuda:{}".format(args.gpu)

    graph, node_group = prepare_data(args)

    dataloader = ClusterDataloader(graph, node_group)

    graph = graph.to(device)

    features = graph.ndata['feat']
    labels = graph.ndata['label']
    train_mask = graph.ndata['train_mask']
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']

    in_feats = graph.ndata['feat'].shape[1]
    hid_feats = args.hid_feats
    out_feats = (graph.ndata['label'].max() + 1).item()

    loss_fcn = torch.nn.CrossEntropyLoss()
    if args.model == 'SGC':
        model = SGC(in_feats=in_feats, hid_feats=hid_feats, out_feats=out_feats)
    else:
        model = SAGE(in_feats=in_feats, hid_feats=hid_feats, out_feats=out_feats)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_global, weight_decay=args.weight_decay)

    best_val = 0

    for epoch in range(args.n_epochs):
        model.train()
        # forward
        logits = model(graph, features) # only compute the train set

        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mic, mac = evaluate(model, graph, features, labels, val_mask)
        print("Epoch {:05d} | Loss {:.4f} | mic {:.4f} | mac {:.4f} |". format(epoch, loss.item(),mic, mac))

        if mic > best_val:
            best_val = mic
            best_epoch = epoch
            save_best_model(model)

        if epoch > best_epoch + 20:
            print(">> Converged at epoch {:d}.".format(epoch))
            break

    model = load_best_model(model)

    mic_f1, mac_f1 = evaluate(model, graph, features, labels, test_mask)
    print(">> Before fine tune: Test mic {:.4f} | Test mac {:.4f}".format(mic_f1, mac_f1))

    local_mics = []

    for i, data in enumerate(dataloader):
        sub_model = model
        loss_fcn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(sub_model.parameters(), lr=0.001,
                                    weight_decay=args.weight_decay)
        sub_train_nid, _, sub_test_nid = data
        for step in range(50):
            sub_model.train()

            logits = sub_model(graph, graph.ndata['feat'])

            loss = loss_fcn(logits[sub_train_nid], labels[sub_train_nid])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        sub_model.eval()

        logits = sub_model(graph, graph.ndata['feat'])
        pred = logits[sub_test_nid]
        sub_test_labels = labels[sub_test_nid]

        local_mic, local_mac = calc_f1(pred.detach().cpu().numpy(), graph.ndata['label'][sub_test_nid].detach().cpu().numpy())

        local_mics.append(local_mic)
        print(">> Subgroup {:d}'s local performance:{:.4f}".format(i, local_mic))

        if i == 0:
            overall_logits = pred.detach().cpu().numpy()
            overall_labels = sub_test_labels.detach().cpu().numpy()
        else:
            overall_logits = np.concatenate((overall_logits, pred.detach().cpu().numpy()), axis=0)
            overall_labels = np.concatenate((overall_labels, sub_test_labels.detach().cpu().numpy()), axis=0)

    test_mic, test_mac = calc_f1(overall_logits, overall_labels)

    print(">> test_mic_f1 {:.6f}, test_mac_f1 {:.6f}".format(test_mic, test_mac))

    with open("\home\wuyao\luozihan\MetaGCN\Experiments\paper\sgc\local_"+args.dataset+"_mic_1.pkl", "wb") as f:
        pickle.dump(local_mics, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SGC')
    parser.add_argument("--gpu", type=int, default=0,
            help="gpu")
    parser.add_argument("--dataset", type=str, default='ogbn')
    parser.add_argument("--lr-global", type=float, default=0.005,
            help="global learning rate")
    parser.add_argument("--lr-adapt", type=float, default=0.005,
            help="adaption learning rate")
    parser.add_argument("--model", type=str, default='SAGE',
            help="SGC or GraphSAGE")
    parser.add_argument("--bias", action='store_true', default=False,
            help="flag to use bias")
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