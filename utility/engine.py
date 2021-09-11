import math
import sys
import time
import torch

import torchvision.models.detection.mask_rcnn

from utility.coco_utils import get_coco_api_from_dataset
from utility.coco_eval import CocoEvaluator
import utility.utils as utils
from torch.nn import CrossEntropyLoss
import numpy as np
from sklearn.metrics import confusion_matrix
from statistics import mean
from segmentation_eval import computeMetrics
from os.path import *


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


def train_one_epoch_fcn_resnet(model, optimizer, data_loader, device, epoch, print_freq):
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
        output = model(images)

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


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.segmentation.FCN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator


def evaluate_fcn(model, data_loader, device, args, epoch):
    """
    This method evaluates FCN model model on dataset in data_loader using device device given
    model, data_loader, device, args, and epoch. Evaluation computes, prints, and saves Mean IoU,
    Mean Frequency Weighted IoU, Mean Accuracy, Pixel Accuracy, Class IoU (Apple Class), and
    Class Mean Accuracy (Apple Class).
    :param model: FCN model
    :param data_loader: data_loader
    :param device: device to run evaluation
    :param args: argument dictionary
    :param epoch: current epoch
    :return:
    """

    model.eval()

    mious = []
    fious = []
    mAcc = []
    pAcc = []
    ious = np.empty((0, 2))
    mAccs = np.empty((0, 2))

    with torch.no_grad():
        for image, targets in data_loader:
            images = torch.stack(list(image), dim=0).to(device)
            targets = torch.stack([t["masks"] for t in targets], dim=0).to(device)

            # Gets prediction for image
            pred = model(images)
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

            # Prints results
            print("Epoch {}\n".format(epoch))
            print("Mean IoU: {}".format(mean(mious)))
            print("Mean frequency weighted IoU: {}".format(mean(fious)))
            print("Mean Accuracy: {}".format(mean(mAcc)))
            print("Pixel Accuracy: {}".format(mean(pAcc)))
            print("Class IoU: {}".format(np.mean(ious, axis=0)))
            print("Class Mean Accuracy: {}".format(np.mean(mAccs, axis=0)))

            # Writes results
            with open(join(args.output_dir, 'train_results.txt'), 'a') as f:
                f.write("Epoch {}\n".format(epoch))
                f.write("Mean IoU: {}\n".format(mean(mious)))
                f.write("Mean Accuracy: {}\n".format(mean(mAcc)))
                f.write("Pixel Accuracy: {}\n".format(mean(pAcc)))
                f.write("Class IoU: {}\n".format(np.mean(ious, axis=0)))
                f.write("Class Mean Accuracy: {}\n".format(np.mean(mAccs, axis=0)))
