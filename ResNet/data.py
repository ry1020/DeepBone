import os

from monai.transforms import AddChannel, ScaleIntensity, RandRotate90, EnsureType, Compose, Resize

from dataset import DatasetFromFolder


def get_roi_dir(data_path):
    output_roi_dir = os.path.join(data_path, "rois")
    return output_roi_dir


def get_strength_file(data_path):
    output_strength_file = os.path.join(data_path, "roi_vm_mean.csv")
    return output_strength_file


# Define transforms
def train_transforms():
    return Compose([
        ScaleIntensity(),
        AddChannel(),
        Resize((256, 256, 256)),
        RandRotate90(),
        EnsureType()])


def test_transforms():
    return Compose([
        ScaleIntensity(),
        AddChannel(),
        Resize((256, 256, 256)),
        EnsureType()])


def target_transform():
    return EnsureType()


def get_training_data(data_path):
    roi_dir = get_roi_dir(data_path)
    train_roi_dir = os.path.join(roi_dir, "train")
    train_strength_file = get_strength_file(data_path).replace(".csv", "_train.csv")

    return DatasetFromFolder(train_roi_dir,
                             train_strength_file,
                             train_transforms(),
                             target_transform())


def get_validation_data(data_path):
    roi_dir = get_roi_dir(data_path)
    valid_roi_dir = os.path.join(roi_dir, "val")
    valid_strength_file = get_strength_file(data_path).replace(".csv", "_val.csv")

    return DatasetFromFolder(valid_roi_dir,
                             valid_strength_file,
                             test_transforms(),
                             target_transform())


def get_inference_data(data_path, inference_subset):
    assert inference_subset in ['train', 'val', 'test']
    roi_dir = get_roi_dir(data_path)
    inference_roi_dir = os.path.join(roi_dir, inference_subset)
    inference_strength_file = get_strength_file(data_path).replace(".csv", ''.join(['_',inference_subset,'.csv']))

    return DatasetFromFolder(inference_roi_dir,
                             inference_strength_file,
                             test_transforms(),
                             target_transform())
                             
