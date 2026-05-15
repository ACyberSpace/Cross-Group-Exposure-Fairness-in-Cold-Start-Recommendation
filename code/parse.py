'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--bpr_batch', type=int,default=1024,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int,default=3,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float,default=0.00025,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int,default=0,
                        help="using the dropout or not")
    parser.add_argument('--testbatch', type=int,default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str,default='ml-100k',
                        help="available datasets: [ml-100k, ml-1m]")
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[10, 20]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int,default=1,
                        help="enable tensorboard")

    parser.add_argument('--comment', type=str,default="lgn")
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--load', type=int,default=0)
    parser.add_argument('--max_epochs', type=int,default=1000)
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--model', type=str, default='lgn', help='rec-model, support [mf, lgn]')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--alpha', type=float, default=0.45, help='weight of fairloss')
    parser.add_argument('--beta', type=float, default=0.35, help='weight of loss')
    parser.add_argument('--theta', type=float, default=1.5, help='Fair re-weighting factor')
    parser.add_argument('--phi', type=float, default=2.0, help='Fair re-weighting factor')
    parser.add_argument('--gamma', type=float, default=3.0, help='Fair re-weighting factor')
    parser.add_argument('--negative_num', type=int, default=1, help='negative sampling number')
    parser.add_argument('--percentile', type=float, default=0.25, help='percentile of tail')
    parser.add_argument('--gat_heads', type=int, default=4, help='heads of gat')
    parser.add_argument('--fair_k', type=int, default=30, help='topk of fairloss')
    parser.add_argument('--sen_attr', type=str, default='Age', help='sensitive attribute to be processed')
    return parser.parse_args()
