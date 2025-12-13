from torchvision import transforms
from torchvision.transforms import Compose
import albumentations as A
import cv2
import numpy as np
from typing import Literal, Optional, Tuple


class AlbumentationsTransform:
    """Wrapper to make Albumentations work with PyTorch DataLoader
    cause 'You have to pass data to augmentations as named arguments, for example: aug(image=image)'
    """

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        img = np.array(img)
        augmented = self.transform(image=img)
        return augmented["image"]


class Transform:
    def __init__(self, img_size):
        self.img_size = img_size

    def get_transforms(
        self, mode: Optional[Literal["augment"]], aug_strength: float = 1.0
    ) -> Tuple:
        """returns train & val transforms"""

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        train_transform = None
        val_transform = A.Compose(
            [
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean, std),
                A.ToTensorV2(),
            ]
        )

        if mode == "augment":
            train_transform = A.Compose(
                [
                    A.Resize(self.img_size, self.img_size),
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=15, p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.05,
                        scale_limit=0.1,
                        rotate_limit=0,
                        border_mode=cv2.BORDER_REFLECT_101,
                        p=0.5,
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=0.3
                    ),
                    A.Normalize(mean=mean, std=std),
                    A.ToTensorV2(),
                ]
            )

        train_wrapped = (
            AlbumentationsTransform(train_transform)
            if train_transform
            else AlbumentationsTransform(val_transform)
        )
        val_wrapped = AlbumentationsTransform(val_transform)

        return (train_wrapped, val_wrapped)
