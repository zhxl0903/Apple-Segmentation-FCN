import math
import sys
from os.path import *
from statistics import mean

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.nn import CrossEntropyLoss

import utility.utils as utils
from segmentation_eval import computeMetrics


def train_one_epoch_fcn(model, optimizer, data_loader, device, epoch, print_freq):
    """
    This method trains FCN model for an epoch given FCN model model, optimizer optimizer,
    data_loader data_loader, device device, epoch epoch, and print frequency print_freq.

    :param model: FCN model
    :param optimizer: optimizer
    :param device: device to run training
    :param epoch: current epoch
    :param print_freq: frequency to print updates from logger
    :return:
    """

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    # Defines softmax cross-entropy loss
    CE_Loss = CrossEntropyLoss()

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = torch.stack(list(images), dim=0).to(device)
        targets = torch.stack([t["masks"] for t in targets], dim=0).long().to(device)

        # Output is the responses before applying softmax activation
        output = model(images)['out']

        loss = CE_Loss(output, targets)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


def evaluate_fcn(model, data_loader, device, args, epoch, dataset='Train'):
    """
    This method evaluates FCN model model on dataset in data_loader using device device given
    model, data_loader, device, args, epoch, and dataset. Evaluation computes, prints, and saves Mean IoU,
    Mean Frequency Weighted IoU, Mean Accuracy, Pixel Accuracy, Class IoU (Apple Class), and
    Class Mean Accuracy (Apple Class).
    :param model: FCN model
    :param data_loader: data_loader
    :param device: device to run evaluation
    :param args: argument dictionary
    :param epoch: current epoch
    :param dataset: dataset (e.g. Train, Val)
    :return:
    """

    model.eval()
    print('Evaluating on {} dataset...'.format(dataset))

    mious = []
    fious = []
    mAcc = []
    pAcc = []
    ious = np.empty((0, 2))
    mAccs = np.empty((0, 2))

    with torch.no_grad():
        for step, (image, targets) in enumerate(data_loader, 1):

            # Prepares batch of images and masks
            images = torch.stack(list(image), dim=0).to(device)
            targets = torch.stack([t["masks"] for t in targets], dim=0).to(device)

            # Gets prediction for image
            pred = model(images)['out']
            pred = pred[0].detach().cpu().numpy()
            pred = np.argmax(pred, axis=0).astype(np.float32)

            # Gets GT mask
            gt_mask = targets[0].detach().cpu().numpy().astype(np.float32)

            # Computes different metrics
            confusion = confusion_matrix(gt_mask.flatten(), pred.flatten())
            miou, fwiou, macc, pacc, iou, maccs = computeMetrics(confusion)
            mious.append(miou)
            fious.append(fwiou)
            mAcc.append(macc)
            pAcc.append(pacc)
            ious = np.vstack((ious, iou))
            mAccs = np.vstack((mAccs, maccs))

            # Prints results from metrics
            print(
                'Epoch: [{}]  [ {}/{}]  Mean IoU: {}  Mean frequency weighted IoU: {}  Mean Accuracy: {}  Pixel Accuracy: {}  Class IoU: {}  Class Mean Accuracy: {}'.format(
                    epoch, step,
                    len(data_loader), miou,
                    fwiou, macc, pacc, iou,
                    maccs))

        # Prints results
        print("Epoch {}".format(epoch))
        print("Dataset: {}".format(dataset))
        print("Mean IoU: {}".format(mean(mious)))
        print("Mean frequency weighted IoU: {}".format(mean(fious)))
        print("Mean Accuracy: {}".format(mean(mAcc)))
        print("Pixel Accuracy: {}".format(mean(pAcc)))
        print("Class IoU: {}".format(np.mean(ious, axis=0)))
        print("Class Mean Accuracy: {}".format(np.mean(mAccs, axis=0)))

        # Writes results
        with open(join(args.output_dir, 'train_results.txt'), 'a') as f:
            f.write("Epoch {}\n".format(epoch))
            f.write("Dataset: {}\n".format(dataset))
            f.write("Mean IoU: {}\n".format(mean(mious)))
            f.write("Mean frequency weighted IoU: {}\n".format(mean(fious)))
            f.write("Mean Accuracy: {}\n".format(mean(mAcc)))
            f.write("Pixel Accuracy: {}\n".format(mean(pAcc)))
            f.write("Class IoU: {}\n".format(np.mean(ious, axis=0)))
            f.write("Class Mean Accuracy: {}\n".format(np.mean(mAccs, axis=0)))
