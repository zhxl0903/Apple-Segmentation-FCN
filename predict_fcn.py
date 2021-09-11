import os
import torch
import torch.utils.data
import torchvision
import numpy as np

from data.apple_dataset import AppleDataset
from torchvision.models.segmentation import fcn_resnet50
from torchvision.models.segmentation import fcn_resnet101

import utility.utils as utils
import utility.transforms as T

import imageio
from os.path import *


######################################################
# Predict with either a FCN-Resnet-50 or FCN-Resnet-101 predictor
# using the MinneApple dataset
######################################################

def get_transform(train):
    transforms = []
    transforms.append(T.CollapseMasks(keepdim=False))
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

    # Checks constraint on num_classes
    if num_classes <= 0 or num_classes > 21:
        raise NotImplementedError('num_classes={} is out of range.'.format(num_classes))

    model = fcn_resnet101(pretrained=False, num_classes=num_classes)

    return model


def main(args):
    num_classes = 2
    device = args.device

    # Load the model from
    print("Loading model")
    # Create the correct model type
    if args.fcn_resnet50:
        model = get_fcn_resnet50_model_instance(num_classes)
    else:
        model = get_fcn_resnet101_model_instance(num_classes)

    # Load model parameters and keep on CPU
    checkpoint = torch.load(args.weight_file, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()

    # Gets list of image dirs
    imgs = list(sorted(os.listdir(os.path.join(args.data_path, "images"))))

    print("Creating data loaders")
    dataset_test = AppleDataset(args.data_path, get_transform(train=False))
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                   shuffle=False, num_workers=1,
                                                   collate_fn=utils.collate_fn)

    # Create output directory
    base_path = os.path.dirname(args.output_path)
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # Predict on mask on each image
    with torch.no_grad():
        for image, targets in data_loader_test:
            im_id = targets[0]['image_id']
            im_name = data_loader_test.dataset.get_img_name(im_id)

            print('Working on {}...'.format(im_name))

            image = torch.stack(list(image), dim=0).to(device)
            output = model(image)

            # Saves mask
            output = output[0].cpu().numpy()
            mask = (np.argmax(output, axis=0) * 255).astype(np.uint8)
            imageio.imsave(join(base_path, im_name), mask)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch Detection')
    parser.add_argument('--data_path', default='/media/zhang205/Datasets/Datasets/MinneApple/detection/test', help='path to the data to predict on')
    parser.add_argument('--output_path', default='/home/zhang205/PycharmProjects/Apple-Segmentation-FCN/results', help='path where to write the prediction outputs')
    parser.add_argument('--weight_file', default='/media/zhang205/Datasets/Datasets/MinneApple/detection/train', help='path to the weight file')
    parser.add_argument('--device', default='cuda', help='device to use. Either cpu or cuda')
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument('--fcn_resnet50', action='store_true', help='use a FCN-Resnet-50 model')
    model.add_argument('--fcn_resnet101', action='store_true', help='use a FCN-Resnet-101 model')

    args = parser.parse_args()
    main(args)
