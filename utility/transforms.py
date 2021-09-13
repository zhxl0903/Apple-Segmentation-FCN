import random
import torch

from torchvision.transforms import functional as F


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    """
    This class defines a composition of transforms.
    """
    def __init__(self, transforms):
        """
        This method initializes an instance of Compose given
        list of transforms transforms
        :param transforms: list of transforms.
        """
        self.transforms = transforms

    def __call__(self, image, target):
        """
        This method applies each transform in self.transforms sequentially to image and
        target given image and target.
        :param image: image
        :param target: target
        :return: image, target after applying each transform in self.transforms
        """
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    """
    This class defines a Random Horizontal Flip transform.
    Image or target data are flipped horizontally with some probability prob.
    """
    def __init__(self, prob):
        """
        This method initializes an instance of RandomHorizontalFlip transform
        given prob.
        :param prob: probability of horizontal flip
        """
        self.prob = prob

    def __call__(self, image, target):
        """
        This method performs random horizontal flip transform on image
        and target with probability self.prob given image and target.
        If a horizontal flip is to be performed and the target masks are
        not available, then only the image is flipped.
        :param image: image
        :param target: target
        :return: image, target
        """
        if random.random() < self.prob:

            # Flips image and GT data, if masks are available.
            # Flips image, otherwise.
            if "masks" in target and target["masks"] is not None:
                height, width = image.shape[-2:]
                image = image.flip(-1)
                bbox = target["boxes"]
                bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
                target["boxes"] = bbox
                if "masks" in target:
                    target["masks"] = target["masks"].flip(-1)
                if "keypoints" in target:
                    keypoints = target["keypoints"]
                    keypoints = _flip_coco_person_keypoints(keypoints, width)
                    target["keypoints"] = keypoints
            else:
                image = image.flip(-1)

        return image, target


class ToTensor(object):
    """
    This class defines a ToTensor transform.
    ToTensor transform converts image to PyTorch Tensor
    and normalizes the image to [0,1] range.
    """
    def __call__(self, image, target):
        """
        This method applies ToTensor transform to image given
        image and target. ToTensor transform converts image to
        Pytorch Tensor and normalizes image to [0,1] range.
        :param image: image
        :param target: target
        :return: image, target
        """
        image = F.to_tensor(image)
        return image, target


class CollapseMasks(object):
    """
        This transform collapses masks in target using OR operation.
    """

    def __init__(self, keepdim=True):
        self.keepdim = keepdim

    def __call__(self, image, target):
        """
        This method collapses masks in target into a binary mask using OR operation given
        image and target.
        :param image: image
        :param target: target dictionary
        :return: target dictionary in which masks are collapsed to a binary mask using OR operation
        """
        if "masks" in target and target["masks"] is not None:
            target['masks'] = target["masks"].any(0, keepdim=self.keepdim)
        return image, target


class ToBinaryClasses(object):
    """
        This transform converts binary masks to 2 masks for classes, background and object
    """

    def __call__(self, image, target):
        """
        This method converts masks in target to 2 classes given image and target.
        :param image: image
        :param target: target dictionary in which masks contain a single binary mask
        :return: target dictionary in which masks contain 2 masks
        """

        if "masks" in target and target["masks"] is not None:
            target["masks"] = torch.cat([(1 - target["masks"]).clone(), target["masks"].clone()], dim=0)
        return image, target
