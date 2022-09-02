import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime


from compute_direction import compute_structure_dir_parallel, compute_mil_dir
from compute_radiomics import compute_radiomics_features_parallel


def get_strength_file(data_path):
    output_strength_file = os.path.join(data_path, "FEA_linear_partial_driver.csv")
    return output_strength_file

def compute_and_combine_features(data_loader, model, data_path, inference_subset):
    print('combine_features')

    inference_strength_file = get_strength_file(data_path).replace(".csv", ''.join(['_',inference_subset,'.csv']))
    img_strengths = pd.read_csv(inference_strength_file)

    model.eval()

    with torch.no_grad():

        for i, (rois, targets) in enumerate(data_loader):
            print('Batch: [{0}/{1}]\t'.format(i + 1,len(data_loader)))
            print('Start Time = {}'.format(datetime.now().time()))

            targets_pred, deeplearning_features = model(rois)

            rois = torch.squeeze(rois)

            rois_list = []
            for j in range(rois.size(dim=0)):
                mil_dir_features_tmp = compute_mil_dir(img_strengths, i*data_loader.batch_size+j) #Make sure data_loader.shuffle is False
                if j == 0:
                    mil_dir_features = mil_dir_features_tmp
                else:
                    mil_dir_features = np.concatenate((mil_dir_features, mil_dir_features_tmp))

                rois_list.append(rois[j,:,:,:].numpy())
            
            structure_dir_features = compute_structure_dir_parallel(rois_list)

            BONE_HU = 1800  
            rois = torch.mul(rois, BONE_HU)   # scaled by BONE_HU which is related to radiomics binWidth

            rois_list = []
            for j in range(rois.size(dim=0)):
                rois_list.append(rois[j,:,:,:].numpy())

            radiomics_features = compute_radiomics_features_parallel(rois_list)

            features_array = np.concatenate((mil_dir_features, structure_dir_features, radiomics_features, deeplearning_features.cpu().numpy()),-1)

            if i == 0:
                features_array_all = features_array
            else:
                features_array_all = np.concatenate((features_array_all, features_array))
            
            print('End Time = {}'.format(datetime.now().time()))
            
    return features_array_all


def load_and_combine_features(data_path, inference_subset, features_path):
    features_file = os.path.join(features_path, ''.join(['features_sim_',inference_subset,'.csv']))
    fatures_all = pd.read_csv(features_file)

    structure_dir_features_name_list = open(os.path.join(features_path, 'all_GST_features_list.txt')).read().split("\n")
    structure_dir_features = fatures_all[structure_dir_features_name_list].to_numpy()

    radiomics_features_name_list = open(os.path.join(features_path, 'all_radiomics_features_list.txt')).read().split("\n")
    radiomics_features = fatures_all[radiomics_features_name_list].to_numpy()

    radiomics_BMD_features = np.expand_dims(fatures_all['original_firstorder_Mean'].to_numpy(), axis=1)     # BMD (original_firstorder_Mean) Only

    deeplearning_features_name_list = open(os.path.join(features_path, 'all_DL_features_list.txt')).read().split("\n")
    deeplearning_features = fatures_all[deeplearning_features_name_list].to_numpy()


    select_features_name_list = open(os.path.join(features_path, 'most14_correlatedwithImportantDL_radiomics_features_list.txt')).read().split("\n")
    select_features = fatures_all[select_features_name_list].to_numpy()
    
    features_array_all = np.concatenate((structure_dir_features, radiomics_features, deeplearning_features),-1) #Choose which features to use for regression

    inference_strength_file = get_strength_file(data_path).replace(".csv", ''.join(['_',inference_subset,'.csv']))
    img_strengths = pd.read_csv(inference_strength_file)
    targets_all = img_strengths.iloc[:, 2]

    if 'name_bone' in fatures_all:
        group = np.expand_dims(fatures_all['name_bone'].to_numpy(), axis=1)
        return features_array_all, targets_all, group
    else:
        return features_array_all, targets_all

