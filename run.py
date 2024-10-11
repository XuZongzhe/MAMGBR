import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["DGLBACKEND"] = "pytorch"
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
import time
from tqdm import tqdm
from model.MAMGBR import MAMGBR
from copy import deepcopy
from model.metric import compute_rr, compute_ndcg, compute_recall
from dataset import DealDataset

def test_model(dataset, model, device, epoch):

    item_total_num = 0
    user_total_num = 0

    ks = [1, 10, 100]
    item_total_rr = [0 for i in range(len(ks))]
    item_total_recall = [0 for i in range(len(ks))]
    item_total_ndcg = [0 for i in range(len(ks))]
    user_total_rr = [0 for i in range(len(ks))]
    user_total_recall = [0 for i in range(len(ks))]
    user_total_ndcg = [0 for i in range(len(ks))]

    for i, data in tqdm(enumerate(dataset)):
        tu, _, _, item_s, user_s = data
        tu = tu.to(device)
        item_s, user_s = item_s.to(device), user_s.to(device)

        loss, item_sample_score, user_sample_score = model(tu, item_s, user_s)

        item_total_num += item_sample_score.shape[0]
        item_rrs = compute_rr(item_sample_score, ks)
        item_recalls = compute_recall(item_sample_score, ks)
        item_ndcgs = compute_ndcg(item_sample_score, ks)

        user_total_num += user_sample_score.shape[0]
        user_rrs = compute_rr(user_sample_score, ks)
        user_recalls = compute_recall(user_sample_score, ks)
        user_ndcgs = compute_ndcg(user_sample_score, ks)

        for i in range(len(ks)):
            item_total_rr[i] += item_rrs[i]
            item_total_recall[i] += item_recalls[i]
            item_total_ndcg[i] += item_ndcgs[i]
            user_total_rr[i] += user_rrs[i]
            user_total_recall[i] += user_recalls[i]
            user_total_ndcg[i] += user_ndcgs[i]

    f = open("log/log_{}_{}.txt".format(args.model, args.remark), "a")
    s0 = "epoch %d ," % epoch
    s2 = "rr@%d:%f" % (ks[1], item_total_rr[1] / item_total_num)
    s3 = "ndcg@%d:%f" % (ks[1], item_total_ndcg[1] / item_total_num)
    s4 = "rr@%d:%f" % (ks[2], item_total_rr[2] / item_total_num)
    s5 = "ndcg@%d:%f" % (ks[2], item_total_ndcg[2] / item_total_num)

    print("Test " + s0 + "\t" + s2 + "\t" + s3 + "\t" + s4 + "\t" + s5)
    f.write(s0 + 'Item Task :' + "\t" + s2 + "\t" + s3 + "\t" + s4 + "\t" + s5 + "\n")

    s2 = "rr@%d:%f" % (ks[1], user_total_rr[1] / user_total_num)
    s3 = "ndcg@%d:%f" % (ks[1], user_total_ndcg[1] / user_total_num)
    s4 = "rr@%d:%f" % (ks[2], user_total_rr[2] / user_total_num)
    s5 = "ndcg@%d:%f" % (ks[2], user_total_ndcg[2] / user_total_num)
    print("Test " + s0 + "\t" + s2 + "\t" + s3 + "\t" + s4 + "\t" + s5)
    f.write(s0 + 'User Task :' + "\t" + s2 + "\t" + s3 + "\t" + s4 + "\t" + s5 + "\n")
    f.close()

    return item_total_ndcg[2] / item_total_num, user_total_ndcg[2] / user_total_num


if __name__ == "__main__":
    if not os.path.exists('log'):
        os.makedirs('log')
    if not os.path.exists("dict"):
        os.makedirs("dict")

    parser = argparse.ArgumentParser(description='MGCNs')
    parser.add_argument('-m', '--model', type=str, default='MTL', help='model')
    parser.add_argument('-s', '--save', type=int, default=0, help='save emb or not')
    parser.add_argument('-g', '--gcn_type', type=str,
                        choices=['lightgcn', 'layergcn', 'imp_gcn', 'ultragcn'], default='lightgcn',
                        help='Choose GCN model')
    args = parser.parse_args()

    args.user_num = 125012
    args.item_num = 30516

    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    in_dimension, hidden_dimension, out_dimension = 128, 16, 10

    model = MAMGBR(in_dimension, hidden_dimension, out_dimension, device, args)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-6)

    trainDataset = DealDataset("train")
    tuneDataset = DealDataset("tune")
    testDataset = DealDataset("test")
    train_loader = DataLoader(dataset=trainDataset, batch_size=64, shuffle=True)
    tune_loader = DataLoader(dataset=tuneDataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=testDataset, batch_size=64, shuffle=True)

    incr = 0
    max_item_rr = 0
    max_user_rr = 0
    epoch = 0

    for epoch in range(30):
        print('epoch %d Start!' % epoch)
        for data in tqdm(train_loader):
            tu, _, _, item_s, user_s = data
            tu = tu.to(device)
            item_s, user_s = item_s.to(device), user_s.to(device)

            loss, item_sample_score, user_sample_score = model(tu, item_s, user_s)
            loss = loss.unsqueeze(1)

            optimizer.zero_grad()
            loss.backward(torch.ones_like(loss))
            optimizer.step()

        end = time.time()
        s0 = "epoch %d " % epoch
        print(s0 + 'have trained')

        if args.save:
            e_pi, e_up, e_ui = deepcopy(model.embed_pi.weight), deepcopy(model.embed_u.weight), deepcopy(
                model.embed.weight)
            np.save('embs/{}_pi_emb_{}.npy'.format(args.model, epoch), e_pi.detach().cpu().numpy())
            np.save('embs/{}_up_emb_{}.npy'.format(args.model, epoch), e_up.detach().cpu().numpy())
            np.save('embs/{}_ui_emb_{}.npy'.format(args.model, epoch), e_ui.detach().cpu().numpy())

        f = open("log/log_{}.txt".format(args.model), "a")
        f.write("tune\n")
        f.close()

        item_rr, user_rr = test_model(tune_loader, model, device, epoch)

        if item_rr > max_item_rr or user_rr > max_user_rr:
            max_item_rr, max_user_rr = item_rr, user_rr
            torch.save(model.state_dict(), "dict/dict_{}.pt".format(args.model))
            print('A better model has been saved')
            incr = 0
        elif item_rr < max_item_rr and user_rr < max_user_rr:
            incr += 1

        t_model = MAMGBR(in_dimension, hidden_dimension, out_dimension, device, args)
        t_model.load_state_dict(torch.load("dict/dict_{}.pt".format(args.model)))
        t_model.to(device)
        f = open("log/log_{}.txt".format(args.model), "a")
        f.write("test\n")
        f.close()
        item_rr, user_rr = test_model(test_loader, t_model, device, epoch)
