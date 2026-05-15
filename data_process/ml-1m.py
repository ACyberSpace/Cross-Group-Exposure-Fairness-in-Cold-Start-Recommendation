# coding=utf-8
from copyreg import pickle
import sys
import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import pickle

import torch

np.random.seed(2022)

dataset = 'ml-100k'
RAW_DATA = './ml-100k'
RATINGS_FILE = os.path.join(RAW_DATA, 'ratings.dat')
USERS_FILE = os.path.join(RAW_DATA, 'users.dat')
ITEMS_FILE = os.path.join(RAW_DATA, 'movies.dat')
# filter_genres = ["Sci-Fi", "Adventure", "Children's", "Crime", "Horror", "Romance"]
# filter_genres = ["Sci-Fi", "Action", "Adventure", "Children's", "Romance", "Musical",
#                  "Animation", "Mystery", "Horror", "Crime", "Thriller", "War"]
filter_genres = ['Sci-Fi', 'Romance', 'Action', 'Crime', 'Adventure', "Musical", "Thriller", "Children's", "Horror"]
filter_genres = sorted(filter_genres)



def format_user_feature(out_file):

    print('format_user_feature', USERS_FILE, out_file)
    user_df = pd.read_csv(USERS_FILE, sep='\t', header=None, engine='python')
    user_df = user_df[[0, 1, 2, 3]]
    user_df.columns = ["u_id_c", 'u_gender_c', 'u_age_c', 'u_occupation_c']

    gender_map = {'M': 0, 'F': 1}
    user_df['u_gender_c'] = user_df['u_gender_c'].map(gender_map).astype('Int64')  # 用 Int64 兼容缺失

    user_df['u_age_c'] = user_df['u_age_c'].apply(
        lambda x: 0 if x < 18 else (x + 5) // 10 - 1 if x < 45 else 4 if x < 50 else 5 if x < 56 else 6)

    # user_df['u_gender_c'] = user_df['u_gender_c'].apply(lambda x: defaultdict(int, {'M': 0, 'F': 1})[x])
    unique_jobs = sorted(user_df['u_occupation_c'].dropna().unique())
    job2id = {job: idx for idx, job in enumerate(unique_jobs)}
    user_df['u_occupation_c'] = user_df['u_occupation_c'].map(job2id).astype('Int64')

    user_df.index = user_df["u_id_c"]
    user_df.index.name = "user_index"
    user_df = user_df.sort_index()
    return user_df

def format_item_feature(out_file):
    print('format_item_feature', ITEMS_FILE, out_file)
    item_df = pd.read_csv(ITEMS_FILE, sep='\t', header=None, engine='python')

    item_df.columns = ["i_id_c", 'i_title_c', 'i_year_c', 'i_genres_c']
    # item_df['i_year_c'] = item_df['i_year_c'].apply(lambda x: int(re.search(r'.*?\(([0-9]+)\)$', x.strip()).group(1)))
    item_df['i_year_c'] = item_df['i_year_c'].apply(
        lambda x: int(re.search(r'\((\d{4})\)', x).group(1)) if isinstance(x, str) and re.search(r'\((\d{4})\)',
                                                                                                 x) else x
    )
    for genre in filter_genres:
        item_df['i_' + genre + '_c'] = item_df['i_genres_c'].apply(lambda x: 1 if x.find(genre) == -1 else 2)

    item_df["genre_sum"] = sum([item_df['i_' + genre + '_c'] for genre in filter_genres])
    item_df = item_df[item_df.genre_sum.isin([10])]

    item_df = item_df.drop(columns=['i_genres_c', 'genre_sum'])
    seps = [0, 1940, 1950, 1960, 1970, 1980, 1985] + list(range(1990, int(item_df['i_year_c'].max()) + 2))
    year_dict = {}
    for i, sep in enumerate(seps[:-1]):
        for j in range(seps[i], seps[i + 1]):
            year_dict[j] = i + 1
    item_df['i_year_c'] = item_df['i_year_c'].apply(lambda x: defaultdict(int, year_dict)[x])
    item_df.index = item_df["i_id_c"]
    item_df.index.name = "item_index"
    for i in range(item_df["i_id_c"].max()):
        if i not in item_df.index:
            item_df.loc[i] = 0
            item_df.loc[i, "i_id_c"] = i
    item_df = item_df.sort_index()
    return item_df

def format_all_inter(out_file, item_df, label01=False):
    print('format_all_inter', RATINGS_FILE, out_file)

    filter_items = []
    for row in item_df.to_dict(orient="records"):
        is_filter = False
        for i, genre in enumerate(filter_genres):
            if row["i_" + genre + "_c"] == 2:
                is_filter = True
                break
        if is_filter == True:
            filter_items.append(row["i_id_c"])

    inter_df = pd.read_csv(RATINGS_FILE, sep='\t', header=None, engine='python')
    inter_df.columns = ["u_id_c", "i_id_c", "label", "time"]
    inter_df = inter_df.sort_values(by=["time", "u_id_c"], kind='mergesort')
    inter_df = inter_df.drop_duplicates(["u_id_c", "i_id_c"]).reset_index(drop=True)
    if label01:
        inter_df["label"] = inter_df["label"].apply(lambda x: 1 if x > 0 else 0)

    inter_df = inter_df[inter_df["i_id_c"].isin(filter_items)]
    return inter_df

def random_split_data(all_data_file, dataset_name, val_size=0.1, test_size=0.2):
    all_data = pd.read_csv(all_data_file, sep='\t', names=['u_id_c', 'i_id_c', 'label', 'time', 'user_freq', 'item_freq'], engine='python')
    user_list = list(all_data["u_id_c"].unique())
    if type(val_size) is float:
        val_size = int(len(all_data) * val_size)
    if type(test_size) is float:
        test_size = int(len(all_data) * test_size)
    validation_set = all_data.sample(n=val_size).sort_index()
    all_data = all_data.drop(validation_set.index)
    test_set = all_data.sample(n=test_size).sort_index()
    train_set = all_data.drop(test_set.index)

    training_dict = {}
    for row in train_set.to_dict(orient="records"):
        if row["u_id_c"] not in training_dict:
            training_dict[row["u_id_c"]] = []
        training_dict[row["u_id_c"]].append((row["i_id_c"], row["label"]))

    validation_dict = {}
    for row in validation_set.to_dict(orient="records"):
        if row["u_id_c"] not in validation_dict:
            validation_dict[row["u_id_c"]] = []
        validation_dict[row["u_id_c"]].append((row["i_id_c"], row["label"]))

    test_dict = {}
    for row in test_set.to_dict(orient="records"):
        if row["u_id_c"] not in test_dict:
            test_dict[row["u_id_c"]] = []
        test_dict[row["u_id_c"]].append((row["i_id_c"], row["label"]))

    for u in user_list:
        if u not in training_dict:
            if u in validation_dict and len(validation_dict[u]) > 1:
                training_dict[u] = []
                training_dict[u].append(validation_dict[u].pop())
            elif u in test_dict and len(test_dict[u]) > 1:
                training_dict[u] = []
                training_dict[u].append(test_dict[u].pop())
        if u not in validation_dict:
            if u in training_dict and len(training_dict[u]) > 1:
                validation_dict[u] = []
                validation_dict[u].append(training_dict[u].pop())
            elif u in test_dict and len(test_dict[u]) > 1:
                validation_dict[u] = []
                validation_dict[u].append(test_dict[u].pop())
        if u not in test_dict:
            if u in training_dict and len(training_dict[u]) > 1:
                test_dict[u] = []
                test_dict[u].append(training_dict[u].pop())
            elif u in validation_dict and len(validation_dict[u]) > 1:
                test_dict[u] = []
                test_dict[u].append(validation_dict[u].pop())

    training_dict_len = sum([len(v) for k, v in training_dict.items()])
    validation_dict_len = sum([len(v) for k, v in validation_dict.items()])
    test_dict_len = sum([len(v) for k, v in test_dict.items()])

    print('train=%d validation=%d test=%d' % (training_dict_len, validation_dict_len, test_dict_len))

    for i, data in enumerate([training_dict, validation_dict, test_dict]):
        user_inter_dict = data
        if i == 0:
            data_path = f"../{dataset}/train.txt"
        elif i == 1:
            data_path = f"../{dataset}/valid.txt"
        elif i == 2:
            data_path = f"../{dataset}/test.txt"

        u_ids = sorted(list(user_inter_dict.keys()))
        with open(data_path, "w") as f:
            for u_id in u_ids:
                for item, rating in user_inter_dict[u_id]:
                    s = str(u_id)
                    s = s + '\t' + str(item) + '\t' + str(rating) + '\n'
                    f.write(s)
                # s += "\n"
                # f.write(s)

    return train_set, validation_set, test_set

def random_split_data_attr_cold(
        all_data_file,
        user_df, item_df,
        cold_user_age_groups, cold_item_genres,
        # === 新增参数 ===
        ku_train=1,  # 每个冷用户在 train 的 few-shot 上限（0=纯0-shot）
        ki_train=1,  # 每个冷物品在 train 的 few-shot 上限
        ratio=(0.7, 0.1, 0.2),  # (train, val, test) 目标比例
        eval_ratio=(1, 2),  # 冷相关样本分配到 (val:test) 的内部比例
        seed=42,
        out_dir="../ml-100k/cold"
):
    """
    冷启动划分（按敏感属性）：
      - 只要交互涉及“冷用户年龄组”或“冷物品类型组”，一律不进训练集，随机分到 val/test
      - 其余交互(热×热)再按比例随机切 val/test，剩余为 train
    不做最小量保底 & 不做三类分层保证量。
    """
    rng = np.random.default_rng(seed)

    all_data = pd.read_csv(
        all_data_file, sep='\t',
        names=['u_id_c', 'i_id_c', 'label', 'time', 'user_freq', 'item_freq'],
        engine='python'
    )

    # 冷用户集合（按 u_age_c 桶编码）
    cold_users = set(
        user_df.loc[user_df['u_age_c'].isin(list(cold_user_age_groups)), 'u_id_c'].astype(int).tolist()
    )
    # 冷物品集合（按主类型字符串 i_genres_c）
    cold_items = set(
        item_df.loc[item_df['i_genres_c'].isin(list(cold_item_genres)), 'i_id_c'].astype(int).tolist()
    )

    # 掩码：涉及冷用户或冷物品
    u_cold = all_data['u_id_c'].isin(cold_users)
    i_cold = all_data['i_id_c'].isin(cold_items)

    A = all_data[(~u_cold) & (~i_cold)].copy()  # 热×热
    B = all_data[(u_cold) & (~i_cold)].copy()  # 冷U×热I
    C = all_data[(~u_cold) & (i_cold)].copy()  # 热U×冷I
    D = all_data[(u_cold) & (i_cold)].copy()  # 冷U×冷I

    # 目标配额
    N = len(all_data)
    rt, rv, rs = ratio
    T_train = int(N * rt / (rt + rv + rs))
    T_val = int(N * rv / (rt + rv + rs))
    T_test = N - T_train - T_val

    train_rows, val_rows, test_rows = [], [], []
    # --- D：冷×冷，全部进 test（如果想部分进 val，可自行再切分）---
    if len(D):
        test_rows.append(D)

    # --- B：冷U×热I，逐用户 few-shot 进 train ---
    if len(B):
        if ku_train > 0:
            B_train_idx = (
                B.groupby('u_id_c', group_keys=False)
                .apply(lambda df: df.sample(n=min(ku_train, len(df)), random_state=int(rng.integers(1e9))))
                .index
            )
            B_train = B.loc[B_train_idx].copy()
            B_left = B.drop(B_train_idx).copy()
            train_rows.append(B_train)
        else:
            B_left = B

        # 剩余按 eval_ratio 切到 val/test
        if len(B_left):
            r_v, r_s = ratio[1:]
            n_val = int(len(B_left) * (r_v / (r_v + r_s)))
            idx = rng.permutation(len(B_left))
            val_rows.append(B_left.iloc[idx[:n_val]])
            test_rows.append(B_left.iloc[idx[n_val:]])

    # --- C：热U×冷I，逐物品 few-shot 进 train ---
    if len(C):
        if ki_train > 0:
            C_train_idx = (
                C.groupby('i_id_c', group_keys=False)
                .apply(lambda df: df.sample(n=min(ki_train, len(df)), random_state=int(rng.integers(1e9))))
                .index
            )
            C_train = C.loc[C_train_idx].copy()
            C_left = C.drop(C_train_idx).copy()
            train_rows.append(C_train)
        else:
            C_left = C

        if len(C_left):
            r_v, r_s = ratio[1:]
            n_val = int(len(C_left) * (r_v / (r_v + r_s)))
            idx = rng.permutation(len(C_left))
            val_rows.append(C_left.iloc[idx[:n_val]])
            test_rows.append(C_left.iloc[idx[n_val:]])

    # --- 先合并目前的三份 ---
    train_set = pd.concat(train_rows, axis=0) if len(train_rows) else pd.DataFrame(columns=all_data.columns)
    val_set = pd.concat(val_rows, axis=0) if len(val_rows) else pd.DataFrame(columns=all_data.columns)
    test_set = pd.concat(test_rows, axis=0) if len(test_rows) else pd.DataFrame(columns=all_data.columns)

    # --- 用 A(热×热) 回填到目标比例 ---
    # 打乱 A
    A = A.sample(frac=1.0, random_state=int(rng.integers(1e9))).reset_index(drop=True)

    need_train = max(0, T_train - len(train_set))
    need_val = max(0, T_val - len(val_set))
    need_test = max(0, T_test - len(test_set))

    a0, a1 = 0, need_train
    train_set = pd.concat([train_set, A.iloc[a0:a1]], axis=0)

    a0, a1 = a1, a1 + need_val
    val_set = pd.concat([val_set, A.iloc[a0:a1]], axis=0)

    a0, a1 = a1, a1 + need_test
    test_set = pd.concat([test_set, A.iloc[a0:a1]], axis=0)

    # 剩余的 A（如果有），按比例丢进 train/val/test（这里简单地全部放 train，也可以再按比例分配）
    if a1 < len(A):
        train_set = pd.concat([train_set, A.iloc[a1:]], axis=0)

    # 打乱并重置 index
    train_set = train_set.sample(frac=1.0, random_state=int(rng.integers(1e9))).reset_index(drop=True)
    val_set = val_set.sample(frac=1.0, random_state=int(rng.integers(1e9))).reset_index(drop=True)
    test_set = test_set.sample(frac=1.0, random_state=int(rng.integers(1e9))).reset_index(drop=True)

    # === 统计并打印 ===
    print(f"[ALL] N={N}; target (train,val,test)=({T_train},{T_val},{T_test})")
    print(f"[NOW] train={len(train_set)}  val={len(val_set)}  test={len(test_set)}")
    if N > 0:
        print("      ratio≈ (%.3f, %.3f, %.3f)" % (len(train_set) / N, len(val_set) / N, len(test_set) / N))

    # === 按你的原格式写盘 ===
    os.makedirs(out_dir, exist_ok=True)
    for i, df in enumerate([train_set, val_set, test_set]):
        path = os.path.join(out_dir, ["train.txt", "valid.txt", "test.txt"][i])
        with open(path, "w") as f:
            for row in df.itertuples(index=False):
                f.write(f"{row.u_id_c}\t{row.i_id_c}\t{row.label}\n")

    # === 同步返回，兼容你原来的返回签名 ===
    return train_set, val_set, test_set

def renumber_ids(df, old_column, new_column):
    old_ids = sorted(df[old_column].dropna().astype(int).unique())
    id_dict = dict(zip(old_ids, range(len(old_ids))))
    id_df = pd.DataFrame({new_column: old_ids, old_column: old_ids})
    id_df[new_column] = id_df[new_column].apply(lambda x: id_dict[x])
    id_df.index = id_df[new_column]
    id_df = id_df.sort_index()
    df[old_column] = df[old_column].apply(lambda x: id_dict[x] if x in id_dict else 0)
    df = df.rename(columns={old_column: new_column})
    return df, id_df, id_dict

def save_structured_embedding(embed_dict, id_list, out_path):
    all_embed = []
    for id_ in id_list:
        attr_vecs = embed_dict[id_]
        # attr_stack = torch.stack(attr_vecs, dim=0)  # [num_attr, embed_dim]
        all_embed.append(attr_vecs)
    all_embed_tensor = torch.stack(all_embed, dim=0)  # [num_id, num_attr, embed_dim]
    torch.save(all_embed_tensor, out_path)
    print(f"Saved {out_path} with shape {tuple(all_embed_tensor.shape)}")


def main():
    data_dir = '../LFM2b/cold'
    file_path_feature = '../Gen_File/'

    item_feature = pickle.load(open(file_path_feature + f'Movie/{dataset}/Item_feature_dict', 'rb'))
    user_feature = pickle.load(open(file_path_feature + f'User/{dataset}/user_feature_dict', 'rb'))

    # user_file = os.path.join(data_dir, 'users.dat')
    # user_df = format_user_feature(user_file)
    #
    # item_file = os.path.join(data_dir, 'items.dat')
    # item_df = format_item_feature(item_file)
    #
    # all_inter_file = os.path.join(data_dir, 'inters.dat')
    # inter_df = format_all_inter(all_inter_file, item_df, label01=False)
    # dataset_name = 'ml1m01-5-1'
    #
    # inter_df['user_freq'] = inter_df.groupby('u_id_c')['u_id_c'].transform('count')
    # inter_df['item_freq'] = inter_df.groupby('i_id_c')['i_id_c'].transform('count')
    #
    # least_freq = 4
    # while np.min(inter_df['user_freq']) <= least_freq:
    #     inter_df.drop(inter_df.index[inter_df['user_freq'] <= least_freq], inplace=True)
    #     inter_df.reset_index(drop=True, inplace=True)
    #     inter_df['item_freq'] = inter_df.groupby('i_id_c')['i_id_c'].transform('count')
    #     inter_df.drop(inter_df.index[inter_df['item_freq'] <= least_freq], inplace=True)
    #     inter_df.reset_index(drop=True, inplace=True)
    #     inter_df['user_freq'] = inter_df.groupby('u_id_c')['u_id_c'].transform('count')
    #     inter_df.reset_index(drop=True, inplace=True)
    #
    # inter_df, uid_df, uid_dict = renumber_ids(inter_df, old_column='u_id_c', new_column="u_id_c")
    # inter_df, iid_df, iid_dict = renumber_ids(inter_df, old_column='i_id_c', new_column="i_id_c")
    # user_df = user_df[user_df["u_id_c"].isin(uid_dict)]
    # item_df = item_df[item_df["i_id_c"].isin(iid_dict)]
    # user_df["u_id_c"] = user_df["u_id_c"].apply(lambda x: uid_dict[x])
    # item_df["i_id_c"] = item_df["i_id_c"].apply(lambda x: iid_dict[x])
    #
    # # 假设 genre columns 是以下列名（你可以按实际情况修改）
    # genre_cols = [col for col in item_df.columns]
    # genre_cols = genre_cols[3:]
    #
    # # 提取类型名称（去掉前缀和后缀）
    # genre_names = [col[2:-2] for col in genre_cols]
    #
    # # 构造新列：Genres_c
    # item_df['Genres_c'] = item_df[genre_cols].apply(
    #     lambda row: '|'.join([genre_names[i] for i, v in enumerate(row) if v == 2]),
    #     axis=1
    # )
    #
    # for genre in filter_genres:
    #     item_df = item_df.drop(columns=[f'i_{genre}_c'])
    #
    # # 合并用户频次列到 user_df
    # user_df = user_df.merge(
    #     inter_df[['u_id_c', 'user_freq']].drop_duplicates('u_id_c'),
    #     on='u_id_c',
    #     how='left'
    # )
    # user_df['user_freq'] = user_df['user_freq'].fillna(0).astype(int)
    #
    # # 合并物品频次列到 item_df
    # item_df = item_df.merge(
    #     inter_df[['i_id_c', 'item_freq']].drop_duplicates('i_id_c'),
    #     on='i_id_c',
    #     how='left'
    # )
    # item_df['item_freq'] = item_df['item_freq'].fillna(0).astype(int)

    # user_attr_keys = sorted(list(next(iter(user_feature.values())).keys()))
    # item_attr_keys = sorted(list(next(iter(item_feature.values())).keys()))

    # save_structured_embedding(user_feature, list(uid_dict.keys()), data_dir+'/user_embedding_tensor.pt')
    # save_structured_embedding(item_feature, list(iid_dict.keys()), data_dir+'/item_embedding_tensor.pt')
    u = range(len(user_feature))
    i = range(len(item_feature))
    save_structured_embedding(user_feature, u, data_dir + '/user_embedding_tensor.pt')
    save_structured_embedding(item_feature, i, data_dir + '/item_embedding_tensor.pt')

    # item_df.to_csv(item_file, index=False, sep='\t', header=False)
    #
    # # item_df = pd.read_csv('../ml-100k/items.dat', header=None, sep='\t', engine='python')
    # item_df.columns = ["i_id_c", 'i_title_c', 'i_year_c', 'i_genres_c', 'i_freq_c']
    # item_group = {}
    # for row in item_df.to_dict(orient="records"):
    #     for i, genre in enumerate(filter_genres):
    #         if row['i_genres_c'] == genre:
    #             item_group[row["i_id_c"]] = i
    # pickle.dump(item_group, open(f"{data_dir}/item_group.pkl", "wb"))
    #
    # user_df.to_csv(user_file, index=False, sep='\t', header=False)
    # # user_df = pd.read_csv('../ml-100k/users.dat', sep='\t', header=None, engine='python')
    # user_df.columns = ["u_id_c", "u_gender_c", "u_age_c", 'u_occupation_c', "u_freq_c"]
    # map_gender = {'M': 0, 'F': 1}
    # user_group = {}
    # for row in user_df.to_dict(orient="records"):
    #     # user_group[row["u_id_c"]] = map_gender[row["u_gender_c"]]
    #     user_group = {int(row["u_id_c"]): int(row["u_age_c"])
    #                   for row in user_df.to_dict(orient="records")}
    #
    # pickle.dump(user_group, open(f"{data_dir}/user_group.pkl", "wb"))
    #
    # inter_df.to_csv(all_inter_file, sep='\t', index=False, header=False)
    # # train_set, _, _ = random_split_data_(all_inter_file, dataset_name, val_size=0.1, test_size=0.2)
    # random_split_data_attr_cold(all_inter_file, user_df, item_df, cold_user_age_groups=[0],
    #                              cold_item_genres=['Adventure'], ratio=(0.7, 0.1, 0.2), out_dir=data_dir)
    # train_set, _, _ = bias_random_split_data(inter_df, user_df, item_df, test_bias_ratio=0.8, val_ratio=0.0, test_ratio=0.2)
    return


# def bias_random_split_data(inter_df, user_df, item_df, test_bias_ratio=0.8, val_ratio=0.0, test_ratio=0.2):
#     # all_data = pd.read_csv(all_data_file, sep='\t', names=['u_id_c', 'i_id_c', 'label', 'time', 'user_freq', 'item_freq'], engine='python')
#     all_data = inter_df.merge(user_df[['u_id_c', 'u_gender_c']], on='u_id_c', how='left')
#     all_data = all_data.merge(item_df[['i_id_c', 'Genres_c']], on='i_id_c', how='left')
#     # 设置测试集偏向类型
#     male_test_genres = {'Romance', 'Musical', "Children's"}
#     female_test_genres = {'Action', 'Sci-Fi', 'Crime', 'Adventure', 'Thriller', 'Horror'}
#
#     def is_preferred(user_gender, genres_str):
#         genres = set(genres_str.split('|'))
#         if user_gender == "M":  # Male
#             return len(genres & male_test_genres) > 0
#         elif user_gender == "F":  # Female
#             return len(genres & female_test_genres) > 0
#         return False
#
#     user_list = list(all_data["u_id_c"].unique())
#     if type(val_ratio) is float:
#         val_size = int(len(all_data) * val_ratio)
#     if type(test_ratio) is float:
#         test_size = int(len(all_data) * test_ratio)
#
#     # 添加标记列：是否为测试偏好
#     all_data['test_biased'] = all_data.apply(lambda row: is_preferred(row['u_gender_c'], row['Genres_c']), axis=1)
#
#     # 随机划分
#     all_data['rand'] = np.random.rand(len(all_data))
#
#     test_prob = np.where(all_data['test_biased'],  0.28, 0.15)
#     test_mask = all_data['rand'] < test_prob
#     val_mask = ~test_mask & (all_data['rand'] < val_ratio)
#     train_mask = ~(test_mask | val_mask)
#
#     train_set = all_data[train_mask].drop(columns=['test_biased', 'rand'])
#     validation_set = all_data[val_mask].drop(columns=['test_biased', 'rand'])
#     test_set = all_data[test_mask].drop(columns=['test_biased', 'rand'])
#
#     training_dict = {}
#     for row in train_set.to_dict(orient="records"):
#         if row["u_id_c"] not in training_dict:
#             training_dict[row["u_id_c"]] = []
#         training_dict[row["u_id_c"]].append((row["i_id_c"], row["label"]))
#
#     validation_dict = {}
#     for row in validation_set.to_dict(orient="records"):
#         if row["u_id_c"] not in validation_dict:
#             validation_dict[row["u_id_c"]] = []
#         validation_dict[row["u_id_c"]].append((row["i_id_c"], row["label"]))
#
#     test_dict = {}
#     for row in test_set.to_dict(orient="records"):
#         if row["u_id_c"] not in test_dict:
#             test_dict[row["u_id_c"]] = []
#         test_dict[row["u_id_c"]].append((row["i_id_c"], row["label"]))
#
#     for u in user_list:
#         if u not in training_dict:
#             if u in validation_dict and len(validation_dict[u]) > 1:
#                 training_dict[u] = []
#                 training_dict[u].append(validation_dict[u].pop())
#             elif u in test_dict and len(test_dict[u]) > 1:
#                 training_dict[u] = []
#                 training_dict[u].append(test_dict[u].pop())
#         if u not in validation_dict:
#             if u in training_dict and len(training_dict[u]) > 1:
#                 validation_dict[u] = []
#                 validation_dict[u].append(training_dict[u].pop())
#             elif u in test_dict and len(test_dict[u]) > 1:
#                 validation_dict[u] = []
#                 validation_dict[u].append(test_dict[u].pop())
#         if u not in test_dict:
#             if u in training_dict and len(training_dict[u]) > 1:
#                 test_dict[u] = []
#                 test_dict[u].append(training_dict[u].pop())
#             elif u in validation_dict and len(validation_dict[u]) > 1:
#                 test_dict[u] = []
#                 test_dict[u].append(validation_dict[u].pop())
#
#     training_dict_len = sum([len(v) for k, v in training_dict.items()])
#     validation_dict_len = sum([len(v) for k, v in validation_dict.items()])
#     test_dict_len = sum([len(v) for k, v in test_dict.items()])
#
#     print('train=%d validation=%d test=%d' % (training_dict_len, validation_dict_len, test_dict_len))
#
#     for i, data in enumerate([training_dict, validation_dict, test_dict]):
#         user_inter_dict = data
#         if i == 0:
#             data_path = "../ml-100k/train.txt"
#         elif i == 1:
#             data_path = "../ml-100k/valid.txt"
#         elif i == 2:
#             data_path = "../ml-100k/test.txt"
#
#         u_ids = sorted(list(user_inter_dict.keys()))
#         with open(data_path, "w") as f:
#             for u_id in u_ids:
#                 for item, rating in user_inter_dict[u_id]:
#                     s = str(u_id)
#                     s = s + '\t' + str(item) + '\t' + str(rating) + '\n'
#                     f.write(s)
#                 # s += "\n"
#                 # f.write(s)
#
#     return train_set, validation_set, test_set

if __name__ == '__main__':
    main()
