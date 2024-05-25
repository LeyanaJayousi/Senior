import numpy as np
import cv2
import torchvision.transforms as T

from pathlib import Path
from torchvision.datasets.vision import VisionDataset
from PIL import Image
from torch.utils.data import DataLoader


def create_shape_mask(shape, image_shape, x1, y1, x2, y2):
    # Initialize a blank mask with the same height and width as the image
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    if shape == 'circle':  # creating segmentation masks according to the shape of the traffic sign
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        radius = min(abs(x2 - x1), abs(y2 - y1)) // 2
        cv2.circle(mask, center, radius, 255, -1)
    elif shape == 'triangle':
        points = np.array([[x1, y2], [x2, y2], [(x1 + x2) // 2, y1]], np.int32)
        cv2.fillPoly(mask, [points], 255)
    elif shape == 'inverse_triangle':
        points = np.array([[x1, y1], [x2, y1], [(x1 + x2) // 2, y2]], np.int32)
        cv2.fillPoly(mask, [points], 255)
    elif shape == 'octagon':
        points = np.array([[x1 + (x2 - x1) * 0.15, y1],
                           [x2 - (x2 - x1) * 0.15, y1],
                           [x2, y1 + (y2 - y1) * 0.3],
                           [x2, y2 - (y2 - y1) * 0.3],
                           [x2 - (x2 - x1) * 0.15, y2],
                           [x1 + (x2 - x1) * 0.15, y2],
                           [x1, y2 - (y2 - y1) * 0.3],
                           [x1, y1 + (y2 - y1) * 0.3]], np.int32)
        cv2.fillPoly(mask, [points], 255)
    elif shape == 'diamond':
        points = np.array([[(x1 + x2) // 2, y1], [x2, (y1 + y2) // 2],
                           [(x1 + x2) // 2, y2], [x1, (y1 + y2) // 2]], np.int32)
        cv2.fillPoly(mask, [points], 255)
    return mask


class SegmentationDataset(VisionDataset):

    def __init__(self,
                 root,
                 image_folder,
                 mask_folder,
                 transforms=None,
                 seed=None,
                 fraction=None,
                 subset=None,
                 image_color_mode="rgb",
                 mask_color_mode="grayscale") -> None:

        super().__init__(root, transforms)  # Initialize the VisionDataset base class
        image_folder_path = Path(self.root) / image_folder
        mask_folder_path = Path(self.root) / mask_folder
        if not image_folder_path.exists():
            raise OSError(f"{image_folder_path} does not exist.")
        if not mask_folder_path.exists():
            raise OSError(f"{mask_folder_path} does not exist.")

        # Validate the color mode for images and masks
        if image_color_mode not in ["rgb", "grayscale"]:
            raise ValueError(
                f"{image_color_mode} is an invalid choice. Please enter from rgb grayscale."
            )
        if mask_color_mode not in ["rgb", "grayscale"]:
            raise ValueError(
                f"{mask_color_mode} is an invalid choice. Please enter from rgb grayscale."
            )

        # Set color modes for images and masks
        self.image_color_mode = image_color_mode
        self.mask_color_mode = mask_color_mode

        if not fraction:  # If no fraction is specified, load all images and masks
            self.image_names = sorted(image_folder_path.glob("*"))
            self.mask_names = sorted(mask_folder_path.glob("*"))
        else:  # Validate the subset input
            if subset not in ["Train", "Test"]:
                raise (ValueError(
                    f"{subset} is not a valid input. Acceptable values are Train and Test."
                ))
            self.fraction = fraction
            self.image_list = np.array(sorted(image_folder_path.glob("*")))
            self.mask_list = np.array(sorted(mask_folder_path.glob("*")))
            if seed:  # If a seed is provided, shuffle the dataset
                np.random.seed(seed)
                indices = np.arange(len(self.image_list))
                np.random.shuffle(indices)
                self.image_list = self.image_list[indices]
                self.mask_list = self.mask_list[indices]
            if subset == "Train":  # Split the dataset into training and testing subsets
                self.image_names = self.image_list[:int(
                    np.ceil(len(self.image_list) * (1 - self.fraction)))]
                self.mask_names = self.mask_list[:int(
                    np.ceil(len(self.mask_list) * (1 - self.fraction)))]
            else:
                self.image_names = self.image_list[
                    int(np.ceil(len(self.image_list) * (1 - self.fraction))):]
                self.mask_names = self.mask_list[
                    int(np.ceil(len(self.mask_list) * (1 - self.fraction))):]

    def __len__(self):  # Return the number of samples in the dataset
        return len(self.image_names)

    # Get the image and mask paths at the specified index
    def __getitem__(self, index):
        image_path = self.image_names[index]
        mask_path = self.mask_names[index]
        with open(image_path, "rb") as image_file, open(mask_path,
                                                        "rb") as mask_file:
            image = Image.open(image_file)
            if self.image_color_mode == "rgb":  # Converting image to the specified color mode
                image = image.convert("RGB")
            elif self.image_color_mode == "grayscale":
                image = image.convert("L")
            mask = Image.open(mask_file)
            if self.mask_color_mode == "rgb":  # Converting mask to the specified color mode
                mask = mask.convert("RGB")
            elif self.mask_color_mode == "grayscale":
                mask = mask.convert("L")
            sample = {"image": image, "mask": mask}
            if self.transforms:  # Apply transforms
                sample["image"] = self.transforms(sample["image"])
                sample["mask"] = self.transforms(sample["mask"])
            return sample


def get_dataloader_single_folder(data_dir2,
                                 image_folder='Images',
                                 mask_folder='Masks',
                                 fraction=0.2,
                                 batch_size=64):

    data_transforms = T.Compose([  # Defining transformations to be applied to both images and masks
        T.ToTensor(),
        T.Resize((40, 40))
        # T.RandomResizedCrop(size=225)
    ])

    image_datasets = {  # Create a dictionary of datasets for 'Train' and 'Test' subsets
        x: SegmentationDataset(data_dir2,
                               image_folder=image_folder,
                               mask_folder=mask_folder,
                               seed=100,
                               fraction=fraction,
                               subset=x,
                               transforms=data_transforms)
        for x in ['Train', 'Test']
    }
    dataloaders = {  # Create a dictionary of dataloaders for 'Train' and 'Test' subsets
        x: DataLoader(image_datasets[x],
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=0)
        for x in ['Train', 'Test']
    }
    return dataloaders
