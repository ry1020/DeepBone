import os
import torch.utils.data as data
import nrrd
import pandas as pd


class DatasetFromFolder(data.Dataset):
    def __init__(self, roi_dir, strength_file, input_transform=None, target_transform=None):
        self.img_strengths = pd.read_csv(strength_file)
        self.roi_dir = roi_dir

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_strengths)

    def __getitem__(self, idx):
        roi_path = os.path.join(self.roi_dir, self.img_strengths.iloc[idx, 0])
        roi, header = nrrd.read(roi_path)
        if self.input_transform:
            roi = self.input_transform(roi)

        strength = self.img_strengths.iloc[idx, 1]

        if self.target_transform:
            strength = self.target_transform(strength)

        return roi, strength

