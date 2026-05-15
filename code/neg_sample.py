from typing import List
import numpy as np
import random

def sample_negative(user_num: int,
                    item_num: int,
                    train_num: int,
                    allPos: List[List[int]],
                    neg_num: int) -> np.ndarray:
    """
    Python 版：仅为有正样本的用户生成 (user, pos, neg_1..neg_K)。
    冷启动用户（无正样本）将被跳过。
    若所有用户均无正样本，返回形状 (0, neg_num+2) 的空数组。
    """
    # 与原逻辑一致：每个用户生成的样本对数量
    perUserNum = (train_num // user_num) if user_num > 0 else 0
    row = neg_num + 2

    # 保护：无有效 perUserNum，或 item_num 非法
    if perUserNum <= 0 or item_num <= 0 or user_num <= 0:
        return np.empty((0, row), dtype=np.int32)

    # 统计有效用户（有正样本）
    valid_users = sum(1 for u in range(user_num) if u < len(allPos) and len(allPos[u]) > 0)
    if valid_users == 0:
        return np.empty((0, row), dtype=np.int32)

    # 预分配输出数组：仅按有效用户数量
    out = np.empty((valid_users * perUserNum, row), dtype=np.int32)

    out_row = 0
    MAX_TRY = 1000

    for user in range(user_num):
        # 取该用户正样本列表
        pos_item = allPos[user] if user < len(allPos) else []
        if not pos_item:
            # 冷启动：跳过
            continue

        # 更快查重
        pos_set = set(pos_item)

        # 极端：若该用户已与全部 item 交互，负样本无法采样
        pos_covers_all = (len(pos_set) >= item_num)

        for _ in range(perUserNum):
            base = out_row

            # 写 user id
            out[base, 0] = user

            # 随机正样本
            pos_idx = random.randrange(len(pos_item))
            out[base, 1] = pos_item[pos_idx]

            # 负采样
            if pos_covers_all:
                # 无法采负样本时全部置 -1
                if neg_num > 0:
                    out[base, 2:2 + neg_num] = -1
            else:
                for k in range(neg_num):
                    tries = 0
                    negitem = -1
                    # 反复尝试直到不在正样本集合中
                    while tries < MAX_TRY:
                        cand = random.randrange(item_num)
                        if cand not in pos_set:
                            negitem = cand
                            break
                        tries += 1
                    out[base, 2 + k] = negitem  # 若超过上限，保持为 -1

            out_row += 1

    # out_row 可能等于 valid_users*perUserNum（正常），也可能更小（理论上不会）
    return out[:out_row]

