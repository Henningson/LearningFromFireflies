import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import cv2
import random

from typing import List


def swap_ff_labels(segmentations):
    # Swap labels
    for i, segmentation in enumerate(segmentations):
        VF_INDEX = 0
        GL_INDEX = 1
        BG_INDEX = 2

        NEW_BG_INDEX = 0
        NEW_VF_INDEX = 1
        NEW_GL_INDEX = 2

        new_seg = np.zeros_like(segmentation)  # Create BG
        new_seg = np.where(segmentation == GL_INDEX, NEW_GL_INDEX, new_seg)
        new_seg = np.where(segmentation == VF_INDEX, NEW_VF_INDEX, new_seg)
        segmentations[i] = new_seg

    return segmentations


class FirefliesDataset(Dataset):
    def __init__(self, path, transform=None):
        self.base_path = path

        image_dir = os.path.join(self.base_path, "images")
        segmentation_dir = os.path.join(self.base_path, "segmentation")

        self.transform = transform

        self._images = self.load_images(image_dir)
        self._segmentations = self.load_images(segmentation_dir)
        self._segmentations = swap_ff_labels(self._segmentations)

    def load_images(self, path) -> List[np.array]:
        image_data = []
        for file in sorted(os.listdir(path)):
            if file.endswith(".png"):
                img_path = os.path.join(path, file)
                image_data.append(cv2.imread(img_path, 0))

        return image_data

    def __len__(self):
        return len(self._images)

    def generate_dataset_fingerprint(self, num_classes=3) -> torch.tensor:
        fingerprint_path = os.path.join(self.base_path, "fingerprint.npy")
        if os.path.isfile(fingerprint_path):
            return torch.from_numpy(np.load(fingerprint_path))

        total_pixels = 0
        class_pixels = [0] * num_classes

        for segmentation in self._segmentations:
            for i in range(num_classes):
                amount_of_pixels = len(np.nonzero(segmentation == i)[0])
                class_pixels[i] += amount_of_pixels
                total_pixels += amount_of_pixels

        class_weights = np.array(
            [class_labels / total_pixels for class_labels in class_pixels]
        )
        np.save(fingerprint_path, class_weights)
        return torch.from_numpy(class_weights)

    def __getitem__(self, index):
        image = self._images[index]
        segmentation = self._segmentations[index]

        if self.transform is not None:
            augmentations = self.transform(image=image, masks=[segmentation])

            image = augmentations["image"]
            segmentation = augmentations["masks"][0]

        return image, segmentation


class FirefliesOnlyGlottis(Dataset):
    def __init__(self, path, transform=None):
        self.base_path = path

        image_dir = os.path.join(self.base_path, "images")
        segmentation_dir = os.path.join(self.base_path, "segmentation")

        self.transform = transform

        self._images = self.load_images(image_dir)
        self._segmentations = self.load_images(segmentation_dir)

        self.generate_dataset_fingerprint()

    def load_images(self, path) -> List[np.array]:
        image_data = []
        for file in sorted(os.listdir(path)):
            if file.endswith(".png"):
                img_path = os.path.join(path, file)
                image_data.append(cv2.imread(img_path, 0))

        return image_data

    def __len__(self):
        return len(self._images)

    def generate_dataset_fingerprint(self, num_classes=3) -> torch.tensor:
        fingerprint_path = os.path.join(self.base_path, "fingerprint.npy")
        if os.path.isfile(fingerprint_path):
            return torch.from_numpy(np.load(fingerprint_path))

        total_pixels = 0
        class_pixels = [0] * num_classes

        for segmentation in self._segmentations:
            for i in range(num_classes):
                amount_of_pixels = len(np.nonzero(segmentation == i)[0])
                class_pixels[i] += amount_of_pixels
                total_pixels += amount_of_pixels

        class_weights = np.array(
            [class_labels / total_pixels for class_labels in class_pixels]
        )
        np.save(fingerprint_path, class_weights)
        return torch.from_numpy(class_weights)

    def __getitem__(self, index):
        image = self._images[index]
        segmentation = self._segmentations[index]
        segmentation[segmentation == 3] = 2
        segmentation[segmentation == 2] = 0

        if self.transform is not None:
            augmentations = self.transform(image=image, masks=[segmentation])

            image = augmentations["image"]
            segmentation = augmentations["masks"][0]

        return image, segmentation


class HLEPlusPlus(Dataset):
    def __init__(self, path, keys, how_many=3, transform=None):
        base_path = path

        image_dirs = [os.path.join(base_path, key, "png/") for key in keys]
        glottal_mask_dirs = [
            os.path.join(base_path, key, "glottal_mask/") for key in keys
        ]
        vocalfold_mask_dirs = [os.path.join(base_path, key, "vf_mask/") for key in keys]
        self._how_many = how_many

        self.transform = transform

        self.images = self.load_from_multiple_dirs(image_dirs)
        self.glottal_masks = self.load_from_multiple_dirs(glottal_mask_dirs)
        self.vocalfold_masks = self.load_from_multiple_dirs(vocalfold_mask_dirs)

    def load_images(self, path) -> List[np.array]:
        image_data = []
        for i, file in enumerate(sorted(os.listdir(path))):
            if i == self._how_many:
                break

            if file.endswith(".png"):
                img_path = os.path.join(path, file)
                image_data.append(cv2.imread(img_path, 0))

        return image_data

    def load_from_multiple_dirs(self, dirs):
        image_data = []
        for dir in dirs:
            image_data += self.load_images(dir)

        return image_data

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        glottal_mask = self.glottal_masks[index]
        vocalfold_mask = self.vocalfold_masks[index]

        segmentation = np.zeros_like(glottal_mask, dtype=np.uint)
        segmentation[vocalfold_mask == 255.0] = 1
        segmentation[glottal_mask == 255.0] = 2

        if self.transform is not None:
            augmentations = self.transform(image=image, masks=[segmentation])

            image = augmentations["image"]
            segmentation = augmentations["masks"][0]

        return image, segmentation.int()


class HLEOnlyGlottis(Dataset):
    def __init__(self, path, keys, how_many=3, transform=None):
        base_path = path

        image_dirs = [os.path.join(base_path, key, "png/") for key in keys]
        glottal_mask_dirs = [
            os.path.join(base_path, key, "glottal_mask/") for key in keys
        ]
        vocalfold_mask_dirs = [os.path.join(base_path, key, "vf_mask/") for key in keys]
        self._how_many = how_many

        self.transform = transform

        self.images = self.load_from_multiple_dirs(image_dirs)
        self.glottal_masks = self.load_from_multiple_dirs(glottal_mask_dirs)
        self.vocalfold_masks = self.load_from_multiple_dirs(vocalfold_mask_dirs)

    def load_images(self, path) -> List[np.array]:
        image_data = []
        for i, file in enumerate(sorted(os.listdir(path))):
            if i == self._how_many:
                break

            if file.endswith(".png"):
                img_path = os.path.join(path, file)
                image_data.append(cv2.imread(img_path, 0))

        return image_data

    def load_from_multiple_dirs(self, dirs):
        image_data = []
        for dir in dirs:
            image_data += self.load_images(dir)

        return image_data

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        glottal_mask = self.glottal_masks[index]
        vocalfold_mask = self.vocalfold_masks[index]

        segmentation = np.zeros_like(glottal_mask, dtype=np.uint)
        # segmentation[vocalfold_mask == 255.0] = 0
        segmentation[glottal_mask == 255.0] = 1

        if self.transform is not None:
            augmentations = self.transform(image=image, masks=[segmentation])

            image = augmentations["image"]
            segmentation = augmentations["masks"][0]

        return image, segmentation.int()


class FFHLE(Dataset):
    def __init__(self, ff_path, hle_path, keys, transform=None):
        image_dir = os.path.join(ff_path, "images")
        segmentation_dir = os.path.join(ff_path, "segmentation")

        self.transform = transform

        ff_images = self.load_ff_images(image_dir)
        ff_segmentations = self.load_ff_images(segmentation_dir)
        ff_segmentations = swap_ff_labels(ff_segmentations)

        image_dirs = [os.path.join(hle_path, key, "png/") for key in keys]
        glottal_mask_dirs = [
            os.path.join(hle_path, key, "glottal_mask/") for key in keys
        ]
        vocalfold_mask_dirs = [os.path.join(hle_path, key, "vf_mask/") for key in keys]

        self.transform = transform
        hle_images = self.load_from_multiple_dirs(image_dirs)
        hle_glottal_masks = self.load_from_multiple_dirs(glottal_mask_dirs)
        hle_vocalfold_masks = self.load_from_multiple_dirs(vocalfold_mask_dirs)

        hle_segmentations = []
        for glottal_mask, vocalfold_mask in zip(hle_glottal_masks, hle_vocalfold_masks):
            segmentation = np.zeros_like(glottal_mask, dtype=np.uint)
            segmentation[vocalfold_mask == 255.0] = 1
            segmentation[glottal_mask == 255.0] = 2

            hle_segmentations.append(segmentation)

        self._images = ff_images + hle_images
        self._segmentations = ff_segmentations + hle_segmentations

    def load_ff_images(self, path) -> List[np.array]:
        image_data = []
        for file in sorted(os.listdir(path)):
            if file.endswith(".png"):
                img_path = os.path.join(path, file)
                image_data.append(cv2.imread(img_path, 0))

        return image_data

    def load_hle_images(self, path) -> List[np.array]:
        image_data = []
        for i, file in enumerate(sorted(os.listdir(path))):
            if file.endswith(".png"):
                img_path = os.path.join(path, file)
                image_data.append(cv2.imread(img_path, 0))

        return image_data

    def load_from_multiple_dirs(self, dirs):
        image_data = []
        for dir in dirs:
            image_data += self.load_hle_images(dir)

        return image_data

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        image = self._images[index]
        segmentation = self._segmentations[index]

        if self.transform is not None:
            augmentations = self.transform(image=image, masks=[segmentation])

            image = augmentations["image"]
            segmentation = augmentations["masks"][0]

        return image, segmentation.int()


class FFHLESingleImage(Dataset):
    def __init__(self, ff_path, hle_path, keys, transform=None):
        image_dir = os.path.join(ff_path, "images")
        segmentation_dir = os.path.join(ff_path, "segmentation")

        self.transform = transform

        ff_images = self.load_ff_images(image_dir)
        ff_segmentations = self.load_ff_images(segmentation_dir)
        ff_segmentations = swap_ff_labels(ff_segmentations)

        image_dirs = [os.path.join(hle_path, key, "png/") for key in keys]
        glottal_mask_dirs = [
            os.path.join(hle_path, key, "glottal_mask/") for key in keys
        ]
        vocalfold_mask_dirs = [os.path.join(hle_path, key, "vf_mask/") for key in keys]

        self.transform = transform
        hle_images = self.load_from_multiple_dirs(image_dirs)
        hle_glottal_masks = self.load_from_multiple_dirs(glottal_mask_dirs)
        hle_vocalfold_masks = self.load_from_multiple_dirs(vocalfold_mask_dirs)

        hle_segmentations = []
        for glottal_mask, vocalfold_mask in zip(hle_glottal_masks, hle_vocalfold_masks):
            segmentation = np.zeros_like(glottal_mask, dtype=np.uint)
            segmentation[vocalfold_mask == 255.0] = 1
            segmentation[glottal_mask == 255.0] = 2

            hle_segmentations.append(segmentation)

        # Let every 12th image be the real image, so that it appears in around every second batch.
        multiply = len(ff_images) // 12
        image_index = random.randint(0, len(hle_images) - 1)
        single_image_list = [hle_images[image_index]] * multiply
        single_segmentation_list = [hle_segmentations[image_index]] * multiply

        self._images = ff_images + single_image_list
        self._segmentations = ff_segmentations + single_segmentation_list

    def load_ff_images(self, path) -> List[np.array]:
        image_data = []
        for file in sorted(os.listdir(path)):
            if file.endswith(".png"):
                img_path = os.path.join(path, file)
                image_data.append(cv2.imread(img_path, 0))

        return image_data

    def load_hle_images(self, path) -> List[np.array]:
        image_data = []
        for i, file in enumerate(sorted(os.listdir(path))):
            if file.endswith(".png"):
                img_path = os.path.join(path, file)
                image_data.append(cv2.imread(img_path, 0))

        return image_data

    def load_from_multiple_dirs(self, dirs):
        image_data = []
        for dir in dirs:
            image_data += self.load_hle_images(dir)

        return image_data

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        image = self._images[index]
        segmentation = self._segmentations[index]

        if self.transform is not None:
            augmentations = self.transform(image=image, masks=[segmentation])

            image = augmentations["image"]
            segmentation = augmentations["masks"][0]

        return image, segmentation.int()


class FFHLEOnlyGlottis(Dataset):
    def __init__(self, ff_path, hle_path, keys, transform=None):
        image_dir = os.path.join(ff_path, "images")
        segmentation_dir = os.path.join(ff_path, "segmentation")

        self.transform = transform

        ff_images = self.load_ff_images(image_dir)
        ff_segmentations = self.load_ff_images(segmentation_dir)

        for i, segmentation in enumerate(ff_segmentations):
            segmentation[segmentation == 2] = 0
            ff_segmentations[i] = segmentation

        image_dirs = [os.path.join(hle_path, key, "png/") for key in keys]
        glottal_mask_dirs = [
            os.path.join(hle_path, key, "glottal_mask/") for key in keys
        ]

        self.transform = transform
        hle_images = self.load_from_multiple_dirs(image_dirs)
        hle_glottal_masks = self.load_from_multiple_dirs(glottal_mask_dirs)

        hle_segmentations = []
        for glottal_mask in hle_glottal_masks:
            segmentation = np.zeros_like(glottal_mask)
            segmentation[glottal_mask > 0] = 1
            hle_segmentations.append(segmentation)

        self._images = ff_images + hle_images
        self._segmentations = ff_segmentations + hle_segmentations

    def load_ff_images(self, path) -> List[np.array]:
        image_data = []
        for file in sorted(os.listdir(path)):
            if file.endswith(".png"):
                img_path = os.path.join(path, file)
                image_data.append(cv2.imread(img_path, 0))

        return image_data

    def load_hle_images(self, path) -> List[np.array]:
        image_data = []
        for i, file in enumerate(sorted(os.listdir(path))):
            if file.endswith(".png"):
                img_path = os.path.join(path, file)
                image_data.append(cv2.imread(img_path, 0))

        return image_data

    def load_from_multiple_dirs(self, dirs):
        image_data = []
        for dir in dirs:
            image_data += self.load_hle_images(dir)

        return image_data

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        image = self._images[index]
        segmentation = self._segmentations[index]

        if self.transform is not None:
            augmentations = self.transform(image=image, masks=[segmentation])

            image = augmentations["image"]
            segmentation = augmentations["masks"][0]

        return image, segmentation.int()


class FFHLEOnlyGlottisSingleImage(Dataset):
    def __init__(self, ff_path, hle_path, keys, transform=None):
        image_dir = os.path.join(ff_path, "images")
        segmentation_dir = os.path.join(ff_path, "segmentation")

        self.transform = transform

        ff_images = self.load_ff_images(image_dir)
        ff_segmentations = self.load_ff_images(segmentation_dir)

        for i, segmentation in enumerate(ff_segmentations):
            segmentation[segmentation == 2] = 0
            ff_segmentations[i] = segmentation

        image_dirs = [os.path.join(hle_path, key, "png/") for key in keys]
        glottal_mask_dirs = [
            os.path.join(hle_path, key, "glottal_mask/") for key in keys
        ]

        self.transform = transform
        hle_images = self.load_from_multiple_dirs(image_dirs)
        hle_glottal_masks = self.load_from_multiple_dirs(glottal_mask_dirs)

        hle_segmentations = []
        for glottal_mask in hle_glottal_masks:
            segmentation = np.zeros_like(glottal_mask)
            segmentation[glottal_mask > 0] = 1
            hle_segmentations.append(segmentation)

        # Let every 12th image be the real image, so that it appears in around every second batch.
        multiply = len(ff_images) // 12
        image_index = random.randint(0, len(hle_images) - 1)
        single_image_list = [hle_images[image_index]] * multiply
        single_segmentation_list = [hle_segmentations[image_index]] * multiply

        self._images = ff_images + single_image_list
        self._segmentations = ff_segmentations + single_segmentation_list

    def load_ff_images(self, path) -> List[np.array]:
        image_data = []
        for file in sorted(os.listdir(path)):
            if file.endswith(".png"):
                img_path = os.path.join(path, file)
                image_data.append(cv2.imread(img_path, 0))

        return image_data

    def load_hle_images(self, path) -> List[np.array]:
        image_data = []
        for i, file in enumerate(sorted(os.listdir(path))):
            if file.endswith(".png"):
                img_path = os.path.join(path, file)
                image_data.append(cv2.imread(img_path, 0))

        return image_data

    def load_from_multiple_dirs(self, dirs):
        image_data = []
        for dir in dirs:
            image_data += self.load_hle_images(dir)

        return image_data

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        image = self._images[index]
        segmentation = self._segmentations[index]

        if self.transform is not None:
            augmentations = self.transform(image=image, masks=[segmentation])

            image = augmentations["image"]
            segmentation = augmentations["masks"][0]

        return image, segmentation.int()
