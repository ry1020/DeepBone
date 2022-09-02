# import necessary libraries
from matplotlib import pyplot as plt
import nrrd
import pandas as pd
import numpy as np
from os import listdir
from clinicalCT_simulation import simulateImage
from compute_direction import compute_structure_dir

from data import get_training_data, get_validation_data
from torch.utils.data import DataLoader
import nibabel as nib


from monai.data import DataLoader, Dataset
from monai.visualize import matshow3d
import os
from monai.transforms import AddChannel, ScaleIntensity, RandRotate90, EnsureType, Compose, Resize



def is_nrrd_file(filename):
    return any(filename.endswith(extension) for extension in [".nrrd"])


def convert_npy_to_csv():
    strength_file = '../data/roi_vm_mean.npy'
    strengths = np.load(strength_file)
    print(strengths.size)

    roi_dir = '../data/rois/all'
    roi_filenames = [x for x in listdir(roi_dir) if is_nrrd_file(x)]
    print(len(roi_filenames))

    df = pd.DataFrame(strengths, roi_filenames)  # convert array into dataframe
    df.to_csv('../DeepBone/data/roi_vm_mean_all.csv', header=False)  # save the dataframe as a csv file


def test_dataloader():
    train_set = get_training_data('/gpfs_projects/ran.yan/Project_Bone/DeepBone/data')
    test_set = get_validation_data('/gpfs_projects/ran.yan/Project_Bone/DeepBone/data')
    training_data_loader = DataLoader(dataset=train_set, num_workers=2, batch_size=2, shuffle=False)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=2, batch_size=2, shuffle=False)
    im, label = train_set[0]
    print(type(im), im.shape, label)
    matshow3d(
        volume=im,
        fig=None, title="input image",
        figsize=(100, 100),
        every_n=10,
        frame_dim=-1,
        show=True,
        cmap="gray",
    )
    train_features, train_labels = next(iter(training_data_loader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    print(train_labels)


def test_monai_example():
    data_path = os.sep.join(["..", "monai-tutorials-main", "3d_classification", "data", "medical", "ixi", "IXI-T1"])
    images = ["IXI314-IOP-0889-T1.nii.gz"]
    images = [os.sep.join([data_path, f]) for f in images]

    img = nib.load(images[0])


def check_img_nrrd():
    roi, header = nrrd.read('/gpfs_projects/ran.yan/Project_Bone/DeepBone/data/Segmentations_Otsu/all/Segmentation-grayscale-9_53-1159.nrrd')
    print(type(roi), roi.shape)
    matshow3d(
        volume=roi,
        fig=None, title="input image",
        # figsize=(100, 100),
        every_n=10,
        frame_dim=-1,
        show=True,
        cmap="gray",
    )


def summarize_MIL():
    img_strengths = pd.read_csv('/gpfs_projects/ran.yan/Project_Bone/DeepBone/data/FEA_linear_partial_driver_L1.csv')
    mil_dir = '/gpfs_projects/ran.yan/Project_Bone/DeepBone/data/Anisotropy_Measurements_L1'
    for i in range(len(img_strengths)):
        mil_filename = img_strengths.iloc[i, 1].replace("Segmentation-grayscale", "ROI")+"-table.csv"
        mil = pd.read_csv(os.path.join(mil_dir, mil_filename))
        mil = mil.rename(columns={"img.nrrd": img_strengths.iloc[i, 1]})
        if i == 0:
            mil_all = mil.drop(columns='Unnamed: 2').T
        else:
            mil_all = pd.concat([mil_all, mil.drop(columns=['rowname','Unnamed: 2']).T])
    mil_all.to_csv('/gpfs_projects/ran.yan/Project_Bone/DeepBone/data/MIL_L1.csv')  # save the dataframe as a csv file


def test_rotate():
    roi, header = nrrd.read('/gpfs_projects/ran.yan/Project_Bone/DeepBone/data/Segmentations_Otsu/all/Segmentation-grayscale-1_53-1159.nrrd')

    roi= simulateImage(roi, 1, 1, (0.156,0.156,0.2), 100)
    # input_transform = Compose([ScaleIntensity(),RandRotate90(prob = 1)])
    input_transform = Compose([ScaleIntensity()])
    roi = input_transform(roi)
    structure_dir_features = compute_structure_dir(roi)
    print(structure_dir_features)


def summarize_BoneJ():
    img_strengths = pd.read_csv('/gpfs_projects/ran.yan/Project_Bone/DeepBone/data/FEA_linear_partial_driver.csv')
    boneJ_dir = '/gpfs_projects/ran.yan/Project_Bone/DeepBone/data/BoneJ_Results'
    for i in range(len(img_strengths)):
        boneJ_filename = img_strengths.iloc[i, 1].replace("Segmentation-grayscale", "ROI")+"-results_table.csv"
        boneJ = pd.read_csv(os.path.join(boneJ_dir, boneJ_filename))
        boneJ = boneJ.rename(columns={"img.nrrd": img_strengths.iloc[i, 1]})
        if i == 0:
            boneJ_all = boneJ.drop(columns=['Unnamed: 0', 'Unnamed: 2']).T
        else:
            boneJ_all = pd.concat([boneJ_all, boneJ.drop(columns=['rowname','Unnamed: 0','Unnamed: 2']).T])
    boneJ_all.to_csv('/gpfs_projects/ran.yan/Project_Bone/DeepBone/data/BoneJ.csv')  # save the dataframe as a csv file


def main():
    summarize_BoneJ()

if __name__ == "__main__":
    main()
