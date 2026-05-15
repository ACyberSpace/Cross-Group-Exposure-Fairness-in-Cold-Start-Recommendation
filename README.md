## XFairRec

This is the Pytorch implementation for our XFair-Rec:

## Introduction

This work streamlines fairness modeling for recommendation. XFair-Rec focuses on the core ingredients—unbiased text semantics, structure de-biasing, and fairness-aware exposure regularization—to achieve cross-group exposure balance while preserving accuracy.

## Enviroment Requirement

`pip install -r requirements.txt`

## Dataset

We provide two datasets: [Movielens-1M](https://drive.google.com/file/d/1K7L0b9gpPWH2wcAmVXT_j0M3mo8mBdP3/view ) and [LastFM-2B](https://drive.google.com/file/d/1K7L0b9gpPWH2wcAmVXT_j0M3mo8mBdP3/view).
see more in `dataloader.py`

For both datasets, we performed the following preprocessing steps:

We removed both user and item nodes with fewer than five interactions. 
For fairness analysis, we grouped users by age and items by type. 
Specifically, for Movielens-1M, we focused on 9 item types: 'Sci-Fi', 'Romance', 'Action', 'Crime', 'Adventure', "Musical", "Thriller", "Children's", and "Horror". 
For LastFM-2B, we selected 4 item types: "rock", "pop", "jazz", and "folk". 
Specific Cold-Start (CS) groups were defined for experimentation: for Movielens-1M, users with age 0 and Adventure items; 
for LastFM-2B, users with age 0 and Folk items.


## An example to run our model

run XFair-Rec on **Movielens-1M** dataset:

* command

`cd code && python main.py --lr=0.001 --dataset="ml-1m" --alpha 0.50 --beta 0.50 --recdim=64`

*NOTE*:

1. If you feel the test process is slow, try to increase the ` testbatch` and enable `multicore`(Windows system may encounter problems with `multicore` option enabled)
2. Use `tensorboard` option, it's good.
3. Since we fix the seed(`--seed=2020` ) of `numpy` and `torch` in the beginning, if you run the command as we do above, you should have the exact output log despite the running time (check your output of *epoch 5* and *epoch 116*).

## Extend:
* If you want to run XFair-Rec on your own dataset, you should go to `dataloader.py`, and implement a dataloader inherited from `BasicDataset`.  Then register it in `register.py`.
* If you want to run your own models on the datasets we offer, you should go to `model.py`, and implement a model inherited from `BasicModel`.  Then register it in `register.py`.
* If you want to run your own sampling methods on the datasets and models we offer, you should go to `Procedure.py`, and implement a function. Then modify the corresponding code in `main.py`

## Project Statement:
Owner：Jun Tang

Institution：School of Cyberspace Security, Jinan University

