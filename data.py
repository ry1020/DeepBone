from os.path import exists, join, basename
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

from dataset import DatasetFromFolder


def get_roi_dir(dest="../DeepBone"):
    output_roi_dir = join(dest, "rois")
    return output_roi_dir


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale_factor),
        ToTensor(),
    ])


# def target_transform(crop_size):
#     return Compose([
#         CenterCrop(crop_size),
#         ToTensor(),
#     ])


def get_training_set(upscale_factor):
    root_dir = get_roi_dir()
    train_dir = join(root_dir, "train")
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(train_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=None)


def get_test_set(upscale_factor):
    root_dir = get_roi_dir()
    test_dir = join(root_dir, "test")
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(test_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=None)
