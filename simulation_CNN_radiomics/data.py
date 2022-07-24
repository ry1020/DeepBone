import os

from monai.transforms import AddChannel, ScaleIntensity, RandRotate90, EnsureType, Compose, Resize

from dataset import DatasetFromFolder


def get_roi_dir(data_path):
    output_roi_dir = os.path.join(data_path, "Segmentations_Otsu", "all")
    return output_roi_dir


def get_strength_file(data_path):
    output_strength_file = os.path.join(data_path, "FEA_linear_partial_driver.csv")
    return output_strength_file


# Define transforms
def train_transforms():
    return Compose([
        AddChannel(),
        ScaleIntensity(),
        #Resize((128,128,128)),
        RandRotate90(),
        EnsureType()])


def test_transforms():
    return Compose([
        AddChannel(),
        ScaleIntensity(),
        #Resize((128,128,128)),
        EnsureType()])


def target_transform():
    return EnsureType()


def get_training_data(data_path, no_sim, noise_scales, resolution_scales, voxel_size_simulated):
    roi_dir = get_roi_dir(data_path)
    train_strength_file = get_strength_file(data_path).replace(".csv", "_train.csv")

    return DatasetFromFolder(roi_dir,
                             train_strength_file,
                             no_sim,
                             noise_scales,
                             resolution_scales,
                             voxel_size_simulated,
                             train_transforms(),
                             target_transform(),
                             seed=None)


def get_validation_data(data_path, no_sim, noise_scales, resolution_scales, voxel_size_simulated):
    roi_dir = get_roi_dir(data_path)
    valid_strength_file = get_strength_file(data_path).replace(".csv", "_val.csv")

    return DatasetFromFolder(roi_dir,
                             valid_strength_file,
                             no_sim,
                             noise_scales,
                             resolution_scales,
                             voxel_size_simulated,
                             test_transforms(),
                             target_transform(),
                             seed=None)


def get_inference_data(data_path, no_sim, noise_scales, resolution_scales, voxel_size_simulated, seed, inference_subset):
    assert inference_subset in ['train', 'val', 'test']
    roi_dir = get_roi_dir(data_path)
    inference_strength_file = get_strength_file(data_path).replace(".csv", ''.join(['_',inference_subset,'.csv']))

    return DatasetFromFolder(roi_dir,
                             inference_strength_file,
                             no_sim,
                             noise_scales,
                             resolution_scales,
                             voxel_size_simulated,
                             test_transforms(),
                             target_transform(),
                             seed)
                             
