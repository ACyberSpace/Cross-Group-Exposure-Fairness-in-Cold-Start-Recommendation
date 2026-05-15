'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
'''
import csv
import os
from collections import defaultdict

import world
import numpy as np
import torch

import utils
from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing






CORES = multiprocessing.cpu_count() // 2


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    
    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset, neg_ratio=neg_k)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    # users = users.repeat_interleave(neg_k)
    # posItems = posItems.repeat_interleave(neg_k)
    # negItems = negItems.reshape(-1)

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.0
    aver_recloss = 0.0
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in tqdm(enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])),
                              total=total_batch,
                              desc=f"epoch {epoch}"):
        total, rec = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += total
        aver_recloss += rec
        if world.tensorboard:
            w.add_scalar(f'TotalLoss', aver_loss, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
            w.add_scalar(f'RecLoss', aver_recloss, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)

    # loss, reg_loss = Recmodel.bpr_loss(users, posItems, negItems)
    # reg_loss = reg_loss * weight_decay
    # loss = loss + reg_loss
    #
    # opt.zero_grad()
    # loss.backward()
    # opt.step()
    aver_loss = aver_loss / total_batch
    aver_recloss = aver_recloss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"TotalLoss{aver_loss:.3f}-RecLoss{aver_recloss:.3f}-{time_info}"


def test_one_batch(X):
    sorted_items = X[0]
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg, hr = [], [], [], []

    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
        hr.append(utils.HitRate_atK(groundTrue, r, k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg),
            'hr':np.array(hr)}

def Test(dataset, Recmodel, epoch, w=None, multicore=0, fair_test=None):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)

    users = list(testDict.keys())

    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks)),
               'hr': np.zeros(len(world.topks)),
    }

    # user_gender_groups, user_inter_groups, item_genre_groups, item_inter_groups = Group_split(dataset)

    # item_exposure = defaultdict(list)

    with torch.no_grad():

        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        # auc_record = []
        # ratings = []
        users_list = []
        rating_list = []
        groundTrue_list = []
        total_batch = len(users) // u_batch_size + 1
        u_batch_size = len(users)

        for batch_users in tqdm(utils.minibatch(users, batch_size=u_batch_size), total=1):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)

            _, rating_K = torch.topk(rating, k=max_K)
            rating_K = rating_K.cpu()
            # for uid, user in enumerate(batch_users):
            #     topk_list = rating_K[uid].tolist()
            #     for rank, item in enumerate(topk_list):
            #         t = 0
            #         if item in groundTrue[uid]:
            #             t = 1
            #         item_exposure[item].append((user, rank, t))
            rating = rating.cpu().numpy()
            del rating
            users_list.extend([u] for u in batch_users)
            rating_list.extend([row.numpy()] for row in rating_K)
            groundTrue_list.extend([row] for row in groundTrue)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size/len(users))

        fair_performance_results = fair_test.evaluate(Recmodel)
        fair_exposure_results = fair_test.evaluate_exposure(Recmodel)

        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
            results['hr'] += result['hr']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        results['hr'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)
        if world.tensorboard:
            w.add_scalars(f'Test/Recall@{world.topks}',
                          {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{world.topks}',
                          {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{world.topks}',
                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/HR@{world.topks}',
                          {str(world.topks[i]): results['hr'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/CV@{world.topks}',
                          {str(world.topks[i]): fair_performance_results['recall_matrix_var'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Min_25@{world.topks}',
                          {str(world.topks[i]): fair_performance_results['recall_matrix_min_25'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/UCV@{world.topks}',
                          {str(world.topks[i]): fair_performance_results['recall_matrix_uside'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/UCV@{world.topks}',
                          {str(world.topks[i]): fair_performance_results['recall_matrix_iside'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/E_UCV@{world.topks}',
                          {str(world.topks[i]): fair_exposure_results['exposure_matrix_uside'][i] for i in
                           range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/E_ICV@{world.topks}',
                          {str(world.topks[i]): fair_exposure_results['exposure_matrix_iside'][i] for i in
                           range(len(world.topks))}, epoch)




        if multicore == 1:
            pool.close()
        print("=== Overall Performance ===")
        print(results)
        print("\n=== Performance Gap ===")
        print(fair_performance_results)
        print("\n=== Exposure Gap ===")
        print(fair_exposure_results)

        return results

