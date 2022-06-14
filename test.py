# import necessary libraries
import os

import monai
import pandas as pd
import numpy as np
from os import listdir
from data import get_training_set, get_test_set
from torch.utils.data import DataLoader
import SimpleITK as sitk


def is_nrrd_file(filename):
    return any(filename.endswith(extension) for extension in [".nrrd"])


def convert_npy_to_csv():
    strength_file = '../DeepBone/data/roi_vm_mean.npy'
    strengths = np.load(strength_file)
    print(strengths.size)

    roi_dir = '../DeepBone/data/rois/all'
    roi_filenames = [x for x in listdir(roi_dir) if is_nrrd_file(x)]
    print(len(roi_filenames))

    df = pd.DataFrame(strengths, roi_filenames)  # convert array into dataframe
    df.to_csv('../DeepBone/data/roi_vm_mean.csv', header=False)  # save the dataframe as a csv file


def test_dataloader():
    train_set = get_training_set()
    test_set = get_test_set()
    training_data_loader = DataLoader(dataset=train_set, num_workers=2, batch_size=2, shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=2, batch_size=2, shuffle=False)
    im, label = train_set[0]
    print(type(im), im.shape, label)


def convert_nrrd_to_nii():
    roi_dir_nrrd = '../DeepBone/data/rois/all'
    roi_dir_nii = '../DeepBone/data/rois/all_nii'
    for roi_filename in listdir(roi_dir_nrrd):
        if is_nrrd_file(roi_filename):
            img = sitk.ReadImage(os.path.join(roi_dir_nrrd, roi_filename))
            sitk.WriteImage(img, os.path.join(roi_dir_nii, roi_filename.replace(".nrrd", ".nii.gz")))


def main():
    test_dataloader()


if __name__ == "__main__":
    main()
