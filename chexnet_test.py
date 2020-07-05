# encoding: utf-8

"""
The main CheXNet model implementation.
"""


import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from read_data import ChestXrayDataSet
from sklearn.metrics import roc_auc_score
from nn_lib import DenseNet121
from utils import compute_AUCs
import sys
CKPT_PATH = sys.argv[1]
OLD_CHECKPOINT = False
N_CLASSES = 14
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
DATA_DIR = './ChestX-ray14/images'
TEST_IMAGE_LIST = './ChestX-ray14/labels/test_list.txt'
BATCH_SIZE = 64


def update_checkpoint_dict(checkpoint):
    import re
    # Code modified from torchvision densenet source for loading from pre .4 densenet weights.
    state_dict = checkpoint['state_dict']
    # Change if you don't want to use nn.DataParallel(model)
    remove_data_parallel = False

    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    for key in list(state_dict.keys()):
        match = pattern.match(key)
        new_key = match.group(1) + match.group(2) if match else key
        new_key = new_key[7:] if remove_data_parallel else new_key
        state_dict[new_key] = state_dict[key]
        # Delete old key only if modified.
        if match or remove_data_parallel:
            del state_dict[key]


def main():

    cudnn.benchmark = True

    # initialize and load the model
    model = DenseNet121(N_CLASSES).cuda()
    model = torch.nn.DataParallel(model).cuda()

    if os.path.isfile(CKPT_PATH):
        print("=> loading checkpoint")
        checkpoint = torch.load(CKPT_PATH)
        if OLD_CHECKPOINT:
            update_checkpoint_dict(checkpoint)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("=> loaded checkpoint")
    else:
        print("=> no checkpoint found")

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    test_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                    image_list_file=TEST_IMAGE_LIST,
                                    transform=transforms.Compose([
                                        transforms.Resize(256),
                                        # crop ten images from original
                                        transforms.TenCrop(224),
                                        transforms.Lambda
                                        (lambda crops: torch.stack(
                                            [transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack(
                                            [normalize(crop) for crop in crops]))
                                    ]))
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=8, pin_memory=True)

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (inp, target) in enumerate(test_loader):
            target = target.cuda()

            # batch_size, num_class
            gt = torch.cat((gt, target), 0)
            # batch_size, n_crops, channels, height, weights
            bs, n_crops, c, h, w = inp.size()
            input_var = torch.autograd.Variable(
                inp.view(-1, c, h, w).cuda(), volatile=True)
            # output shape: batch_size*n_crops,num_class
            output = model(input_var)
            # average across crops
            # output_mean shape: batch_size, num_class
            output_mean = output.view(bs, n_crops, -1).mean(1)
            pred = torch.cat((pred, output_mean.data), 0)

    # np.save("./pred.pkl", pred.cpu().numpy())
    # np.save("./gt.pkl", gt.cpu().numpy())

    AUROCs = compute_AUCs(gt, pred, N_CLASSES)
    AUROCs = np.array(AUROCs)
    AUROC_avg = AUROCs.mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs[i]))
    base_dir = os.path.dirname(CKPT_PATH)
    np.savetxt(os.path.join(base_dir, "test_auroc.txt"),AUROCs)


if __name__ == '__main__':
    main()
