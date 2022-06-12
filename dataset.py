import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image
import numpy as np
import nrrd

import torch
import torchvision
import torchvision.transforms as transforms

VOX_SIZE = (0.05, 0.05, 0.05)  # mm
BONE_HU = 1800

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".nrrd"])


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y


class DatasetFromFolder(data.Dataset):
    def __init__(self, roi_dir):
        super(DatasetFromFolder, self).__init__()
        self.roi_filenames = [join(roi_dir, x) for x in listdir(roi_dir) if is_image_file(x)]
        self.strength_filename =
    def __getitem__(self, index):
        roi, header = nrrd.read(self.roi_filenames[index])
        roi = torch.tensor(roi)
        roi = (roi - torch.min(roi)) / (torch.max(roi) - torch.min(roi))
        roi = roi * BONE_HU

        roi_vm_mean = np.load(self.strength_filename)  # This is mean stress.
        target = roi_vm_mean[index]

        return roi, target

    def __len__(self):
        return len(self.image_filenames)