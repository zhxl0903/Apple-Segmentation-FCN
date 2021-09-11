import os
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image


#####################################
# Class that takes the input instance masks
# and extracts bounding boxes on the fly
#####################################
class AppleDataset(data.Dataset):
    """
    This class defines the Apple Dataset. Apple Dataset contains
    image, bounding box positions, segmentation masks, image_id,
    and bounding box areas.
    """
    def __init__(self, root_dir, transforms=None):
        """
        This method initializes an instance of AppleDataset given dataset dir
        root_dir and transforms.

        :param root_dir: root dir of AppleDataset which contains images and masks dir
        :param transforms: transforms on data
        """
        self.root_dir = root_dir
        self.transforms = transforms

        # Load all image and mask files, sorting them to ensure they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root_dir, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root_dir, "masks"))))

    def __getitem__(self, idx):

        # Loads images and masks
        img_path = os.path.join(self.root_dir, "images", self.imgs[idx])
        mask_path = os.path.join(self.root_dir, "masks", self.masks[idx])

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)  # Each color of mask corresponds to a different instance with 0 being the background

        # Converts the PIL image to np array
        mask = np.array(mask)
        obj_ids = np.unique(mask)

        # Removes background id
        obj_ids = obj_ids[1:]

        # Splits the color-encoded masks into a set of binary masks
        masks = mask == obj_ids[:, None, None]

        # Gets bbox coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        h, w = mask.shape
        for ii in range(num_objs):
            pos = np.where(masks[ii])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            if xmin == xmax or ymin == ymax:
                continue

            xmin = np.clip(xmin, a_min=0, a_max=w)
            xmax = np.clip(xmax, a_min=0, a_max=w)
            ymin = np.clip(ymin, a_min=0, a_max=h)
            ymax = np.clip(ymax, a_min=0, a_max=h)
            boxes.append([xmin, ymin, xmax, ymax])

        # Converts everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # There is only one class (apples)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # All instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def get_img_name(self, idx):
        return self.imgs[idx]


class TransformDataset(torch.utils.data.Dataset):
    """
        Given a dataset, this class creates a dataset which applies a transform
        to the given dataset.
    """

    def __init__(self, dataset, transforms=None):
        """
        This method initializes an instance of TransformDataset given dataset
        and transforms.
        :param dataset: dataset
        :param transforms: transforms
        """
        self.dataset = dataset
        self.transforms = transforms

    def __getitem__(self, index):
        if self.transforms is not None:
            return self.transforms(self.dataset[index])
        else:
            return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
