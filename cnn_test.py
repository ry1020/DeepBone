import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import glob
import nrrd
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    # Size of the test split
    num_bones_test = 7 # number of bones reserved for testing
    test_split_size = num_bones_test/16
    num_test = int(num_bones_test*13)

    fem_dir = "../BoneBox/data/"
    roi_vm_mean = np.load(fem_dir+"roi_vm_mean.npy")  # This is mean stress.

    # image parameters
    voxSize = (0.05, 0.05, 0.05) # mm
    boneHU = 1800

    # get ROIs
    roiDir = "../BoneBox/data/rois/"

    def apply_scaling(roi):
        roi = (roi - np.min(roi)) / (np.max(roi) - np.min(roi))
        roi = roi * boneHU
        return roi

    def get_roi(number):
        filename_nrrd = glob.glob(roiDir + f"*_roi_{number}.nrrd")[0]
        roi, header = nrrd.read(filename_nrrd)
        roi = apply_scaling(roi)
        return roi

    numROIs = 208

    roi.append = get_roi(0)
    print(type(roi))

    # Splitting data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split_size, random_state=3, shuffle=False)
    #for rind in range(numROIs):