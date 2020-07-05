import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from read_data import ChestXrayDataSet, ChestXrayDataSetWithAugmentation, ChestXrayDataSetWithAugmentationEachEpoch
from sklearn.metrics import roc_auc_score
from nn_lib import DenseNet121
from utils import compute_AUCs
import pickle
import os
import yaml
from time import time
from utils import *
import sys

N_CLASSES = 14
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
DATA_DIR = './ChestX-ray14/images'
TRAIN_IMAGE_LIST = './ChestX-ray14/labels/train_list.txt'
DEV_IMAGE_LIST = './ChestX-ray14/labels/val_list.txt'
TEST_IMAGE_LIST = './ChestX-ray14/labels/test_list.txt'
PRINT_FREQ = 5


params = {
    "train_batch_size": 32,
    "test_batch_size": 64,
    "lr": 1e-4,
    "beta": 0,
    "epochs": 25,
    "base_dir": "./Apr26-aug-epoch",
    "aug_dataset": True,
}
if len(sys.argv) > 1:
    params["base_dir"] = sys.argv[1]

if os.path.exists(params["base_dir"]):
    assert False

params["model_file_path"] = os.path.join(params["base_dir"], "epoch-%d.pkl")
params["best_model_file_path"] = os.path.join(
    params["base_dir"], "best_model.pkl")
params["history_file_path"] = os.path.join(params["base_dir"], "history.pkl")
params["params_file_path"] = os.path.join(params["base_dir"], "params.yml")


def main():
    cudnn.benchmark = True
    model = DenseNet121(N_CLASSES).cuda()
    model = torch.nn.DataParallel(model).cuda()

    # data preprocess
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
                                        transforms.Resize(256),
                                        # crop ten images from original
                                        transforms.TenCrop(224),
                                        transforms.Lambda
                                        (lambda crops: torch.stack(
                                            [transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack(
                                            [normalize(crop) for crop in crops]))
                                        ])

    # load data
    # train_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
    #                                  image_list_file=TRAIN_IMAGE_LIST,
    #                                  transform=train_transform)
    # train_loader = DataLoader(
    #     dataset=train_dataset, batch_size=params["train_batch_size"], shuffle=True, num_workers=8, pin_memory=True)
    train_evaluation_dataset = ChestXrayDataSet(
        data_dir=DATA_DIR, image_list_file=TRAIN_IMAGE_LIST, transform=test_transform)
    train_evaluation_loader = DataLoader(
        dataset=train_evaluation_dataset, batch_size=params["test_batch_size"], shuffle=False, num_workers=8, pin_memory=True)
    dev_dataset = ChestXrayDataSet(
        data_dir=DATA_DIR, image_list_file=DEV_IMAGE_LIST, transform=test_transform)
    dev_loader = DataLoader(
        dataset=dev_dataset, batch_size=params["test_batch_size"], shuffle=False, num_workers=8, pin_memory=True)
    test_dataset = ChestXrayDataSet(
        data_dir=DATA_DIR, image_list_file=TEST_IMAGE_LIST, transform=test_transform)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=params["test_batch_size"], shuffle=False, num_workers=8, pin_memory=True)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["lr"], weight_decay=params["beta"])

    def train(epoch, dev_auc):
        print("start training epoch %d" % (epoch))
        start_time = time()
        local_step = 0
        running_loss = 0
        running_loss_list = []
        model.train()
        if dev_auc is not None:
            train_dataset = ChestXrayDataSetWithAugmentationEachEpoch(
                data_dir=DATA_DIR, image_list_file=TRAIN_IMAGE_LIST, aucs=dev_auc, transform=train_transform)
        else:
            train_dataset = ChestXrayDataSet(
                data_dir=DATA_DIR, image_list_file=TRAIN_IMAGE_LIST, transform=train_transform)
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=params["train_batch_size"], shuffle=True, num_workers=8, pin_memory=True)
        for i, (inp, target) in enumerate(train_loader):
            inp = inp.cuda()
            target = target.cuda()
            optimizer.zero_grad()
            output = model(inp)
            local_loss = F.binary_cross_entropy(output, target)
            running_loss += local_loss.item()
            local_loss.backward()
            optimizer.step()
            if (i+1) % PRINT_FREQ == 0:
                running_loss /= PRINT_FREQ
                print("epoch %d, batch %d/%d, loss: %.5f" %
                      (epoch, i+1, len(train_loader), running_loss))
                running_loss_list.append(running_loss)
                running_loss = 0
        print("end training epoch %d, time elapsed: %.2fmin" %
              (epoch, (time()-start_time)/60))
        return dict(running_loss_list=running_loss_list)

    def evaluate(epoch, dataset_loader, pytorch_dataset, dataset_name):
        print("start evaluating epoch %d on %s" % (epoch, dataset_name))
        gt = torch.tensor([], dtype=torch.float32, device="cuda")
        pred = torch.tensor([], dtype=torch.float32, device="cuda")
        loss = 0.
        model.eval()
        with torch.no_grad():
            for i, (inp, target) in enumerate(dataset_loader):
                target = target.cuda()
                gt = torch.cat((gt, target), 0)
                bs, n_crops, c, h, w = inp.size()
                inp_reshaped = inp.view(-1, c, h, w).cuda()
                output = model(inp_reshaped)
                output_mean = output.view(bs, n_crops, -1).mean(1)
                pred = torch.cat((pred, output_mean), 0)
                local_loss = F.binary_cross_entropy(output_mean, target)
                loss += local_loss*len(target)/len(pytorch_dataset)
        AUROCs = compute_AUCs(gt, pred, N_CLASSES)
        AUROC_avg = np.array(AUROCs).mean()
        print("epoch %d, %s, loss: %.5f, avg_AUC: %.5f" %
              (epoch, dataset_name, loss, AUROC_avg))
        print("epoch %d, %s, individual class AUC" % (epoch, dataset_name))
        for i in range(N_CLASSES):
            print('\tthe AUROC of %s is %.5f' % (CLASS_NAMES[i], AUROCs[i]))
        return dict(auroc=dict(zip(CLASS_NAMES, AUROCs)), auroc_avg=AUROC_avg, loss=loss.item())

    def init_history():
        return dict(epoch=0, train_eval_vals_list=[], dev_eval_vals_list=[],
                    best_dev_eval_vals=dict(auroc_avg=-np.inf, loss=np.inf), best_dev_eval_vals_epoch=-1)

    def update_history(history, epoch, train_eval_vals, dev_eval_vals):
        history["epoch"] = epoch
        history["train_eval_vals_list"].append(train_eval_vals)
        history["dev_eval_vals_list"].append(dev_eval_vals)
        if dev_eval_vals["auroc_avg"] > history["best_dev_eval_vals"]["auroc_avg"]:
            history["best_dev_eval_vals"] = dev_eval_vals
            history["best_dev_eval_vals_epoch"] = epoch
            if epoch >= 1:
                print("saving model...")
                state_dict = model.state_dict()
                torch.save(state_dict, params["best_model_file_path"])
        if epoch >= 1:
            state_dict = model.state_dict()
            torch.save(state_dict, params["model_file_path"] % (epoch))
        with open(params["history_file_path"], 'wb') as f:
            pickle.dump(history, f)

    def train_initialization():
        if not os.path.exists(params["base_dir"]):
            os.mkdir(params["base_dir"])
        with open(params["params_file_path"], 'w') as f:
            yaml.dump(params, f, default_flow_style=False)
        if os.path.exists(params["history_file_path"]):
            with open(params["history_file_path"], 'rb') as f:
                old_history = pickle.load(f)
                last_epoch = old_history["epoch"]
                if last_epoch > params["epochs"]:
                    print("training completed")
                    exit(0)
                model_file = params["model_file_path"] % (last_epoch)
                if os.path.exists(model_file):
                    model.load_state_dict(torch.load(model_file))
                    print("training resumed from epoch %d" % last_epoch)
                    return old_history, last_epoch
        return init_history(), 0

    history, last_epoch = train_initialization()
    for epoch in range(last_epoch+1, params["epochs"]+1):
        if len(history["dev_eval_vals_list"]) == 0:
            dev_roc = None
        else:
            dev_roc = [history["dev_eval_vals_list"][-1]["auroc"][class_name]
                       for class_name in CLASS_NAMES]
        train_eval_vals = train(epoch, dev_roc)
        train_eval_vals2 = evaluate(
            epoch, train_evaluation_loader, train_evaluation_dataset, "train set")
        dev_eval_vals = evaluate(epoch, dev_loader, dev_dataset, "dev set")
        update_history(history, epoch, train_eval_vals, dev_eval_vals)
    print("training completed")


if __name__ == "__main__":
    # dev_auc = np.linspace(1, 0.5, 14)
    # train_transform = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.RandomCrop(224),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406],
    #                          [0.229, 0.224, 0.225]),
    # ])
    # train_dataset = ChestXrayDataSetWithAugmentationEachEpoch(
    #     data_dir=DATA_DIR, image_list_file=TRAIN_IMAGE_LIST, aucs=dev_auc, transform=train_transform)
    main()
