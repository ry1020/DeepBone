import numpy as np
from radiomics import featureextractor
import SimpleITK as sitk
import functools

    
def compute_radiomic_features(volume, settings=None):
    # Define settings for signature calculation
    # These are currently set equal to the respective default values
    
    # Initialize feature extractor
    setting_path='/gpfs_projects/ran.yan/Project_Bone/DeepBone/data/pyradiomics_settings.yaml'
    extractor = featureextractor.RadiomicsFeatureExtractor(setting_path)  
    
    # Extract radiomics from volume
    volumeSITK = sitk.GetImageFromArray(volume)
    mask = np.ones(volume.shape).astype(int)
    mask[0,0,0] = 0 # TODO: this is a temporary fix https://github.com/AIM-Harvard/pyradiomics/issues/765
    maskSITK = sitk.GetImageFromArray(mask)
    
    # featureVector = extractor.computeFeatures(volumeSITK, maskSITK, imageTypeName="original")
    featureVector = extractor.execute(volumeSITK, maskSITK)
    featureVector = {key: value for key, value in featureVector.items() if "diagnostics_" not in key} # TODO: output diagnostics too. remove diagnostic entries, leave only features
    featureVectorArray = np.array([featureVector[featureName].item() for featureName in featureVector.keys()])
    featureNamesList = [featureName for featureName in featureVector.keys()]
    
    return featureVectorArray


def compute_radiomics_features_parallel(volumeList, settings=None, numWorkers=None):
    # Extract radiomics features from a list of volumes using the same settings.
    # https://stackoverflow.com/questions/60116458/multiprocessing-pool-map-attributeerror-cant-pickle-local-object
    #
    # Note: The output featureVectorArray is in (Nvolumes, Nfeatures)
    
    from multiprocessing import cpu_count, Pool
    
    if numWorkers is None:
        numWorkers = cpu_count() - 2

    with Pool(numWorkers) as pool:
        featureVectors = pool.map(functools.partial(compute_radiomic_features, settings=settings),volumeList)
    
    featureVectorArray = np.vstack(featureVectors)

    return featureVectorArray