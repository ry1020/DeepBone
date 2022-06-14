from os.path import join

from monai.transforms import AddChannel, ScaleIntensity, RandRotate90, EnsureType, Compose, Resize

from dataset import DatasetFromFolder
# from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize


def get_roi_dir(dest="../DeepBone/data"):
    output_roi_dir = join(dest, "rois")
    return output_roi_dir


def get_strength_file(dest="../DeepBone/data"):
    output_strength_file = join(dest, "roi_vm_mean.csv")
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


def get_training_set():
    roi_dir = get_roi_dir()
    train_dir = join(roi_dir, "train")
    strength_file = get_strength_file()

    return DatasetFromFolder(train_dir,
                             strength_file,
                             train_transforms(),
                             target_transform=None)


def get_test_set():
    roi_dir = get_roi_dir()
    test_dir = join(roi_dir, "test")
    strength_file = get_strength_file()

    return DatasetFromFolder(test_dir,
                             strength_file,
                             test_transforms(),
                             target_transform=None)

