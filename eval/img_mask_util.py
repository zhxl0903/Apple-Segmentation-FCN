import os
import numpy as np
from PIL import Image
import cv2


class ImageMaskUtil:
    # RGB Color Maps for labelling
    label_color_map = {
        'background': (0, 0, 0),  # background
        'apple': (224, 0, 224),  # apple
    }

    def __init__(self, img_dir=None, mask_dir=None, transforms=None):
        print("***", img_dir, '\n', mask_dir)
        if not img_dir or not os.path.exists(img_dir):
            raise FileNotFoundError("Image path does not exist")
        if not mask_dir or not os.path.exists(mask_dir):
            raise FileNotFoundError("Mask path does not exist")

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transforms = transforms

        # Load all image and mask files, sorting them to ensure they are aligned
        file_types = ("png", "jpg", "jpeg")
        self.imgs = list(sorted(os.listdir(img_dir)))
        self.imgs = [i for i in self.imgs if i.endswith(file_types)]

        self.masks = list(sorted(os.listdir(mask_dir)))
        self.masks = [i for i in self.masks if i.endswith(file_types)]
        print("***", self.imgs, '\n', self.masks)

        if len(self.imgs) != len(self.masks):
            raise ValueError("Number of images must be equal to number of masks")

    @staticmethod
    def overlay_mask_on_image(image: np.array, mask: np.array, alpha=1.0, beta=0.9, gamma=0.0) -> np.array:
        """
        alpha = 1  # transparency for the original image
        beta = 0.9  # transparency for the segmentation map
        gamma = 0  # scalar added to each sum
        """
        # mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return cv2.addWeighted(image, alpha, mask, beta, gamma)

    def __getitem__(self, idx):
        pass

    def get_img_mask_overlay(self, idx, n_channels=3, cls="apple"):
        if idx >= self.__len__():
            raise IndexError(f"requested image index {idx} must be less than {self.__len__()}")

        # Load image and mask
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        img: Image.Image = Image.open(img_path).convert("RGB")
        img_res = np.array(img)

        mask: Image.Image = Image.open(mask_path)
        # Convert the PIL image to np array
        mask_np = np.array(mask)
        # Convert non-zero values to constant 1, retains 0 (background)
        mask_np = np.minimum(1, mask_np)

        mask_col = []
        for i in range(0, n_channels):
            res = np.multiply(mask_np, np.full_like(mask_np, self.label_color_map[cls][i], dtype=np.uint8))
            mask_col.append(res)
        mask_res = np.stack(mask_col, axis=2)

        return Image.fromarray(ImageMaskUtil.overlay_mask_on_image(img_res, mask_res))

    def __len__(self):
        return len(self.imgs)

    def get_img_name(self, idx):
        return self.imgs[idx]

    def show_image(self, idx):
        if idx >= self.__len__():
            raise IndexError(f"requested image index {idx} must be less than {self.__len__()}")
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        img: Image.Image = Image.open(img_path).convert("RGB")
        img.show()

    def show_mask(self, idx):
        if idx >= self.__len__():
            raise IndexError(f"requested mask index {idx} must be less than {self.__len__()}")
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        mask: Image.Image = Image.open(mask_path)
        mask.show()
