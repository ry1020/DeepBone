import os
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, CenterCrop


from dataset import DatasetFromFolder



def get_roi_dir(data_path):
    output_roi_dir = os.path.join(data_path, "Segmentations_Otsu_All")
    return output_roi_dir


def get_strength_file(data_path):
    output_strength_file = os.path.join(data_path, "FEA_linear_partial_driver.csv")
    return output_strength_file


# Define transforms
def train_transforms():
    return Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def test_transforms():
    return Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def target_transform():
    return None


def get_training_data(data_path, no_sim, noise_scales, resolution_scales, voxel_size_simulated, train_subset):
    roi_dir = get_roi_dir(data_path)
    train_strength_file = get_strength_file(data_path).replace(".csv", ''.join(['_',train_subset,'.csv']))

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
                             
