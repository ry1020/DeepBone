import os

import torch
from monai.transforms import AddChannel, ScaleIntensity, RandRotate90, EnsureType, Compose, Resize

from dataset import DatasetFromFolder


def get_roi_dir(dest=os.path.join("..", "data")):
    output_roi_dir = os.path.join(dest, "rois")
    return output_roi_dir


def get_strength_file(dest=os.path.join("..", "data")):
    output_strength_file = os.path.join(dest, "roi_vm_mean.csv")
    return output_strength_file


# Define transforms
def train_transforms():
    return Compose([
        ScaleIntensity(),
        AddChannel(),
        Resize((256, 256, 256)),
        RandRotate90(),
        EnsureType()])


def test_transforms():
    return Compose([
        ScaleIntensity(),
        AddChannel(),
        Resize((256, 256, 256)),
        EnsureType()])


def target_transform():
    return EnsureType()


def get_training_set():
    roi_dir = get_roi_dir()
    train_roi_dir = os.path.join(roi_dir, "train")
    train_strength_file = get_strength_file().replace(".csv", "_train.csv")

    return DatasetFromFolder(train_roi_dir,
                             train_strength_file,
                             train_transforms(),
                             target_transform())


def get_validating_set():
    roi_dir = get_roi_dir()
    valid_roi_dir = os.path.join(roi_dir, "val")
    valid_strength_file = get_strength_file().replace(".csv", "_val.csv")

    return DatasetFromFolder(valid_roi_dir,
                             valid_strength_file,
                             test_transforms(),
                             target_transform())


def get_inference_set():
    roi_dir = get_roi_dir()
    inference_roi_dir = os.path.join(roi_dir, "test")
    inference_strength_file = get_strength_file().replace(".csv", "_test.csv")

    return DatasetFromFolder(inference_roi_dir,
                             inference_strength_file,
                             test_transforms(),
                             target_transform())
