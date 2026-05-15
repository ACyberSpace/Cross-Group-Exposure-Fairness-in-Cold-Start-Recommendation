import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed, reproducibility=True)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset
from evaluator import Fairness_Evaluator

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
fair_test = Fairness_Evaluator(path='../data/' + world.dataset, dataloader=dataset, dname=world.dataset)
fair_test.load_data()

bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cuda')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")

Neg_k = world.negative_num
best_recall = 0.0
patience = world.patience
count = 0
best_epoch = 0


# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
        join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        # if epoch > 50 and epoch % 5 == 0:
        if epoch % 5 == 0:
            cprint("[TEST]")
            result = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'], fair_test)
            if result['recall'][1] < best_recall:
                count += 1
                cprint(f'[PATIENCE] {count}/{patience}')
            else:
                count = 0
                best_epoch = epoch
                best_recall = result['recall'][1]
                cprint("The best model is saved")
                torch.save(Recmodel.state_dict(), weight_file)

            if count >= patience:
                cprint(f"Early stopping at epoch {epoch}, best recall: {best_recall}, best epoch: {best_epoch}")
                break


        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        torch.save(Recmodel.state_dict(), weight_file)
finally:
    if world.tensorboard:
        w.close()