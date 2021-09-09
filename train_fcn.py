import datetime
import os
import time

import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.segmentation import fcn_resnet50
from torchvision.models.segmentation import fcn_resnet101

from data.apple_dataset import AppleDataset
from utility.engine import train_one_epoch, evaluate

import utility.utils as utils
import utility.transforms as T

######################################################
# Train either a Faster-RCNN or Mask-RCNN predictor
# using the MinneApple dataset
######################################################

model_urls = {
    'fcn_resnet50_coco': 'https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth',
    'fcn_resnet101_coco': 'https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth',
    'deeplabv3_resnet50_coco': 'https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth',
    'deeplabv3_resnet101_coco': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth',
}

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_fcn_resnet50_model_instance(num_classes):
    """
    This method gets an instance of FCN-Resnet-50 model given num_classes.
    Model is pretrained on a subset of COCO train2017, on the 20 categories
    that are present in the Pascal VOC dataset. Output layer of model
    uses the first num_classes pretrained filters. 0 < num_classes <= 21

    :param num_classes: number of classes
    :return: instance of FCN-Resnet-50 model.
    """

    # Checks constraint on num_classes
    if num_classes <= 0 or num_classes > 21:
        raise NotImplementedError('num_classes={} is out of range.'.format(num_classes))

    model = fcn_resnet50(pretrained=False, num_classes=num_classes)

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
    return model

def get_fcn_resnet101_model_instance(num_classes):
    """
    This method gets an instance of FCN-Resnet-101 model given num_classes.
    Model is pretrained on a subset of COCO train2017, on the 20 categories
    that are present in the Pascal VOC dataset. Output layer of model
    uses the first num_classes pretrained filters. 0 < num_classes <= 21

    :param num_classes: number of classes
    :return: instance of FCN-Resnet-101 model.
    """
    model = fcn_resnet101(pretrained=False, num_classes=num_classes)

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
    return model

def get_maskrcnn_model_instance(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model


def get_frcnn_model_instance(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def main(args):
    print(args)
    device = args.device

    # Data loading code
    print("Loading data")
    num_classes = 2
    dataset = AppleDataset(os.path.join(args.data_path, 'train'), get_transform(train=True))
    dataset_test = AppleDataset(os.path.join(args.data_path, 'test'), get_transform(train=False))

    print("Creating data loaders")
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.workers, collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                   shuffle=False, num_workers=args.workers,
                                                   collate_fn=utils.collate_fn)

    print("Creating model")
    # Create the correct model type
    if args.model == 'maskrcnn':
        model = get_maskrcnn_model_instance(num_classes)
    else:
        model = get_frcnn_model_instance(num_classes)

    # Move model to the right device
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    #  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    print("Start training")
    start_time = time.time()
    for epoch in range(args.epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq)
        lr_scheduler.step()

        if args.output_dir:
            torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            },  os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))

        # evaluate after every epoch
        evaluate(model, data_loader_test, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Detection Training')
    parser.add_argument('--data_path', default='/home/nicolai/phd/data/cvppp_2019/apple_dataset', help='dataset')
    parser.add_argument('--dataset', default='AppleDataset', help='dataset')
    parser.add_argument('--model', default='maskrcnn', help='model: fcn_resnet50, fcn_resnet101')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=2, type=int)
    parser.add_argument('--epochs', default=13, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.02, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[8, 11], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    args = parser.parse_args()
    print(args.model)
    assert(args.model in ['fcn_resnet50', 'fcn_resnet101'])

    if args.output_dir:
        utils.mkdir(args.output_dir)

    main(args)
