# import necessary libraries
import pandas as pd
import numpy as np
from os import listdir

from data import get_training_set, get_validating_set
from torch.utils.data import DataLoader
import nibabel as nib

from monai.data import DataLoader, Dataset
from monai.visualize import matshow3d
import os


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
    # TODO: Add (image_name,strength) at the first line
    df.to_csv('../data/roi_vm_mean_all.csv', header=False)  # save the dataframe as a csv file


def test_dataloader():
    train_set = get_training_set()
    val_set = get_validating_set()
    training_data_loader = DataLoader(dataset=train_set, num_workers=2, batch_size=2, shuffle=False)
    val_data_loader = DataLoader(dataset=val_set, num_workers=2, batch_size=2, shuffle=False)
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
    im, label = train_set[1]
    print(type(im), im.shape, label)
    train_features, train_labels = next(iter(training_data_loader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    print(train_labels)


def test_monai_example():
    data_path = os.sep.join(["..", "monai-tutorials-main", "3d_classification", "data", "medical", "ixi", "IXI-T1"])
    images = ["IXI314-IOP-0889-T1.nii.gz"]
    images = [os.sep.join([data_path, f]) for f in images]

    img = nib.load(images[0])


def main():
    convert_npy_to_csv()


if __name__ == "__main__":
    main()

