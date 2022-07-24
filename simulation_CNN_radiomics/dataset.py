import os
import numpy as np
import torch.utils.data as data
import nrrd
import pandas as pd
from torch import squeeze

from clinicalCT_simulation import simulateImage


class DatasetFromFolder(data.Dataset):
    def __init__(self, roi_dir, strength_file, no_sim, noise_scales, resolution_scales, voxel_size_simulated, input_transform=None, target_transform=None, seed=None):
        super(DatasetFromFolder, self).__init__()

        self.img_strengths = pd.read_csv(strength_file)
        self.roi_dir = roi_dir
        self.no_sim = no_sim
        self.noise_scales = noise_scales
        self.resolution_scales = resolution_scales
        self.voxel_size_simulated = voxel_size_simulated
        self.seed = seed

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_strengths)

    def __getitem__(self, idx):
        roi_path = os.path.join(self.roi_dir, self.img_strengths.iloc[idx, 1])
        roi, header = nrrd.read(''.join((roi_path,'.nrrd')))

        #array_1 = roi[:,:,np.shape(roi)[2]//2]

        if not self.no_sim:
            roi= simulateImage(roi, self.noise_scales, self.resolution_scales, self.voxel_size_simulated, self.seed)

        #array_2 = roi[:,:,np.shape(roi)[2]//2]

        if self.input_transform:
            roi = self.input_transform(roi)

        #array_3 = squeeze(roi[:,:,:,roi.size(dim=3)//2]).numpy()

        strength = self.img_strengths.iloc[idx, 2]

        strength = np.float32(strength)

        if self.target_transform:
            strength = self.target_transform(strength)

        return roi, strength

