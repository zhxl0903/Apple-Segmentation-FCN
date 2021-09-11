import datetime
import os
import time
from os.path import *

import torch
import torch.utils.data
from torch.utils.data import random_split

import torchvision
from torchvision.models.segmentation import fcn_resnet50
from torchvision.models.segmentation import fcn_resnet101
import numpy as np
import random

from data.apple_dataset import AppleDataset, TransformDataset
from utility.engine import train_one_epoch_fcn_resnet, evaluate_fcn

import utility.utils as utils
import utility.transforms as T

######################################################
# Train either a FCN-Resnet-50 or FCN-Resnet-101 predictor
# using the MinneApple dataset
######################################################

# For reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

model_urls = {
    'fcn_resnet50_coco': 'https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth',
    'fcn_resnet101_coco': 'https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth',
    'deeplabv3_resnet50_coco': 'https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth',
    'deeplabv3_resnet101_coco': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth',
}


def get_transform(train):
    transforms = []
    transforms.append(T.CollapseMasks(keepdim=False))
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def get_fcn_resnet50_model_instance(num_classes, pretrained=True):
    """
    This method gets an instance of FCN-Resnet-50 model given num_classes, pretrained.
    If pretrained, then the model is pretrained on a subset of COCO train2017, on the
    20 categories that are present in the Pascal VOC dataset. Output layer of model
    uses the first num_classes pretrained filters. 0 < num_classes <= 21

    :param num_classes: number of classes
    :param pretrained: loads pretrained model iff pretrained
    :return: instance of FCN-Resnet-50 model.
    """

    # Checks constraint on num_classes
    if num_classes <= 0 or num_classes > 21:
        raise NotImplementedError('num_classes={} is out of range.'.format(num_classes))

    model = fcn_resnet50(pretrained=False, num_classes=num_classes)

    if not pretrained:
        return model

    arch_type = 'fcn'
    backbone = 'resnet50'

    # Loads pretrained model with num_classes number of classes
    arch = arch_type + '_' + backbone + '_coco'
    model_url = model_urls[arch]
    if model_url is None:
        raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
    else:

        # Loads state dictionary of pretrained model
        state_dict = torch.hub.load_state_dict_from_url(model_url, progress=True)

        # gets state dict of model
        state_dict_self = model.state_dict()
        for i, (name, param) in enumerate(state_dict_self.items()):
            if name in ['classifier.4.weight', 'classifier.4.bias']:
                state_dict_self[name].copy_(state_dict[name][:num_classes])
                # print(torch.all(model.state_dict()[name] == state_dict[name][:num_classes]))
            else:
                state_dict_self[name].copy_(state_dict[name])
                # print(torch.all(model.state_dict()[name] == state_dict[name]))
    print('Loaded pretrained model')
    return model


def get_fcn_resnet101_model_instance(num_classes, pretrained=True):
    """
    This method gets an instance of FCN-Resnet-101 model given num_classes.
     If pretrained, then the model is pretrained on a subset of COCO train2017,
    on the 20 categories that are present in the Pascal VOC dataset. Output layer
    of model uses the first num_classes pretrained filters. 0 < num_classes <= 21

    :param num_classes: number of classes
    :param pretrained: loads pretrained model iff pretrained
    :return: instance of FCN-Resnet-101 model.
    """

    # Checks constraint on num_classes
    if num_classes <= 0 or num_classes > 21:
        raise NotImplementedError('num_classes={} is out of range.'.format(num_classes))

    model = fcn_resnet101(pretrained=False, num_classes=num_classes)

    if not pretrained:
        return model

    arch_type = 'fcn'
    backbone = 'resnet101'

    # Loads pretrained model with num_classes number of classes
    arch = arch_type + '_' + backbone + '_coco'
    model_url = model_urls[arch]
    if model_url is None:
        raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
    else:

        # Loads state dictionary of pretrained model
        state_dict = torch.hub.load_state_dict_from_url(model_url, progress=True)

        # gets state dict of model
        state_dict_self = model.state_dict()
        for i, (name, param) in enumerate(state_dict_self.items()):
            if name in ['classifier.4.weight', 'classifier.4.bias']:
                state_dict_self[name].copy_(state_dict[name][:num_classes])
                # print(torch.all(model.state_dict()[name] == state_dict[name][:num_classes]))
            else:
                state_dict_self[name].copy_(state_dict[name])
                # print(torch.all(model.state_dict()[name] == state_dict[name]))
    print('Loaded pretrained model.')
    return model


def main(args):
    print(args)
    device = args.device

    # Data loading code
    print("Loading data")
    num_classes = 2

    # Creates AppleDataset from train set
    dataset = AppleDataset(os.path.join(args.data_path, 'train'), transforms=None)

    # Splits train dataset into train, val set
    data_len = len(dataset)
    data_train_val_len = int(data_len * args.val_percent)
    train, val = random_split(dataset, [data_len - data_train_val_len, data_train_val_len])

    # Gets dataset with transforms
    train = TransformDataset(train, transforms=get_transform(train=True))
    val = TransformDataset(val, transforms=get_transform(train=False))

    print("Creating data loaders")
    data_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.workers, collate_fn=utils.collate_fn)

    data_loader_val = torch.utils.data.DataLoader(val, batch_size=1,
                                                  shuffle=False, num_workers=args.workers,
                                                  collate_fn=utils.collate_fn)

    print("Creating model")

    # Create the correct model type
    if args.model == 'fcn_resnet50':
        model = get_fcn_resnet50_model_instance(num_classes, pretrained=args.pretrained)
    else:
        model = get_fcn_resnet101_model_instance(num_classes, pretrained=args.pretrained)

    # Moves model to the right device
    model = model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]

    # Creates the correct optimizer
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Creates the correct learning rate scheduler
    if args.scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    else:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    # Loads model if args.resume
    if args.resume:
        checkpoint = torch.load(args.resume_model_dir, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    # Clears train results file
    with open(join(args.output_dir, 'train_results.txt'), 'w') as f:
        f.write('')

    print("Start training")
    start_time = time.time()
    for epoch in range(args.epochs):

        # Trains model for one epoch
        train_one_epoch_fcn_resnet(model, optimizer, data_loader, device, epoch, args.print_freq)

        # Updates scheduler
        lr_scheduler.step()

        # Saves model of epoch epoch
        if args.output_dir:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
            }, os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)), _use_new_zipfile_serialization=False)

        # evaluates model on train set
        evaluate_fcn(model, data_loader, device=device, args=args, epoch=epoch, dataset='Train')

        # evaluates model on validation set
        evaluate_fcn(model, data_loader_val, device=device, args=args, epoch=epoch, dataset='Val')

    # Computes total time elapsed
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch Detection Training')
    parser.add_argument('--data_path', default='/media/zhang205/Datasets/Datasets/MinneApple/detection/train', help='dataset')
    parser.add_argument('--dataset', default='AppleDataset', help='dataset')
    parser.add_argument('--val_percent', default=0.1, type=float, metavar='V', help='percent of train set for validation split')
    parser.add_argument('--model', default='fcn_resnet50', help='model: fcn_resnet50, fcn_resnet101')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='loads pretrained model iff pretrained')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=2, type=int)
    parser.add_argument('--epochs', default=16, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 16)')
    parser.add_argument('--optim', default='adam', help='optimizer: adam, sgd')
    parser.add_argument('--lr', default=0.02, type=float, help='initial learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, metavar='b1', help='adam beta 1')
    parser.add_argument('--beta2', default=0.999, type=float, metavar='b2', help='adam beta 2')
    parser.add_argument('--epsilon', default=1e-8, type=float, metavar='eps', help='adam epsilon for numerical stability')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='sgd momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay', dest='weight_decay')
    parser.add_argument('--scheduler', default='MultiStepLR', help='learning rate scheduler: StepLR, MultiStepLR')
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs for StepLR scheduler')
    parser.add_argument('--lr-steps', default=[8, 11], nargs='+', type=int, help='decrease lr at epoch lr-steps for MultiStepLR scheduler')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='./models', help='path where to save')
    parser.add_argument('--resume', dest='resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--resume_model_dir', default='', help='path of model to load to resume training')

    args = parser.parse_args()
    print(args.model)
    assert (args.model in ['fcn_resnet50', 'fcn_resnet101'])

    if args.output_dir:
        utils.mkdir(args.output_dir)

    main(args)
