import nrrd
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from monai.transforms import AddChannel, ScaleIntensity, RandRotate90, EnsureType, Compose, Resize
from PIL import Image as im
from sklearn.preprocessing import StandardScaler

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

from clinicalCT_simulation import simulateImage


def hierarchy_plot():
    dataframe = pd.read_csv('C:/Users/Ran.Yan/Workspace/Project_Bone/DeepBone_Data/features_sim/features_sim_train.csv')
    dataframe = dataframe.iloc[:, 2:]
    correlations = dataframe.corr()
    plt.figure(figsize=(15,10))
    sns.heatmap(round(correlations,2), cmap='bwr')


    plt.figure(figsize=(12,5))
    dissimilarity = 1 - abs(correlations)
    Z = linkage(squareform(dissimilarity), 'complete')

    dendrogram(Z, labels=dataframe.columns, orientation='top', leaf_rotation=90);

    # Clusterize the data
    threshold = 0.8
    labels = fcluster(Z, threshold, criterion='distance')


    # Keep the indices to sort labels
    labels_order = np.argsort(labels)

    # Build a new dataframe with the sorted columns
    for idx, i in enumerate(dataframe.columns[labels_order]):
        if idx == 0:
            clustered = pd.DataFrame(dataframe[i])
        else:
            df_to_append = pd.DataFrame(dataframe[i])
            clustered = pd.concat([clustered, df_to_append], axis=1)
    # clustered.to_csv('C:/Users/Ran.Yan/Workspace/Project_Bone/DeepBone_Data/features_sim/features_sim_train_clusterfeatures_2.csv')
    
    correlations = clustered.corr()
    # correlations.to_csv('C:/Users/Ran.Yan/Workspace/Project_Bone/DeepBone_Data/features_sim/features_sim_train_clusteredcorrelation_2.csv')
    sns.heatmap(round(correlations,2), cmap='bwr')


def hierarchy_plot_2():
    import scipy
    import pylab
    import scipy.cluster.hierarchy as sch

    # Generate features and distance matrix.
    # x = scipy.rand(40)
    # D = scipy.zeros([40,40])
    # for i in range(40):
    #     for j in range(40):
    #         D[i,j] = abs(x[i] - x[j])
    D = pd.read_csv('C:/Users/Ran.Yan/Workspace/Project_Bone/DeepBone_Data/features_sim/features_sim_train.csv')
    D = D.to_numpy()
    

    # Compute and plot dendrogram.
    fig = pylab.figure()
    axdendro = fig.add_axes([0.09,0.1,0.2,0.8])
    Y = sch.linkage(D, method='centroid')
    Z = sch.dendrogram(Y, orientation='right')
    axdendro.set_xticks([])
    axdendro.set_yticks([])

    # Plot distance matrix.
    axmatrix = fig.add_axes([0.3,0.1,0.6,0.8])
    index = Z['leaves']
    D = D[index,:]
    D = D[:,index]
    im = axmatrix.matshow(D, aspect='auto', origin='lower')
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])

    # Plot colorbar.
    axcolor = fig.add_axes([0.91,0.1,0.02,0.8])
    pylab.colorbar(im, cax=axcolor)

    # Display and save figure.
    fig.show()
    fig.savefig('dendrogram.png')


def select_features():
    features_name_txt = open("C:/Users/Ran.Yan/Workspace/Project_Bone/DeepBone_Data/features_sim/most_20_correlated_radiomics_features_list.txt", "r")
    file_content = features_name_txt.read()
    content_list = file_content.split("\n")
    dataframe = pd.read_csv('C:/Users/Ran.Yan/Workspace/Project_Bone/DeepBone_Data/features_sim/features_sim_train.csv')
    dataframe = dataframe[content_list]

def plot_corr():
    correlations = pd.read_csv('C:/Users/Ran.Yan/Workspace/Project_Bone/DeepBone_Data/features_sim/features_sim_train_clusteredcorrelation_only_important.csv', index_col = 0)
    plt.figure(figsize=(30,10))
    sns.heatmap(round(correlations,2), vmax=1, vmin=-1, cmap='bwr')

def plot_roi():
    roi_uCT, header = nrrd.read('C:/Users/Ran.Yan/Workspace/Project_Bone/DeepBone_Data/ROI_grayscale-4_762-68335-L45.nrrd')
    # array_11 = roi_uCT[np.shape(roi_uCT)[0]//2,:,:]
    # plt.figure()
    # imgplot = plt.imshow(array_11, cmap='gray')

    # array_12 = roi_uCT[:,np.shape(roi_uCT)[1]//2,:]
    # plt.figure()
    # imgplot = plt.imshow(array_12, cmap='gray')

    # array_13 = roi_uCT[:,:,np.shape(roi_uCT)[2]//2]
    # plt.figure()
    # imgplot = plt.imshow(array_13, cmap='gray')

    roi, header = nrrd.read('C:/Users/Ran.Yan/Workspace/Project_Bone/DeepBone_Data/Segmentations_Otsu_All/Segmentation-grayscale-4_762-68335-L45.nrrd')

    # array_21 = roi[np.shape(roi)[0]//2,:,:]
    # plt.figure()
    # imgplot = plt.imshow(array_21, cmap='gray')

    # array_22 = roi[:,np.shape(roi)[1]//2,:]
    # plt.figure()
    # imgplot = plt.imshow(array_22, cmap='gray')

    # array_23 = roi[:,:,np.shape(roi)[2]//2]
    # plt.figure()
    # imgplot = plt.imshow(array_23, cmap='gray')

    roi_simulated= simulateImage(roi, noise_scales = 1, resolution_scales =1, voxel_size_simulated =(0.156,0.156,0.2), seed=100)
    
    roi_simulated_rotated = np.rot90(np.flip(roi_simulated,1))

    # array_31 = roi_simulated_rotated[np.shape(roi_simulated_rotated)[0]//2,:,:]
    # plt.figure()
    # imgplot = plt.imshow(array_31, cmap='gray')

    # array_32 = roi_simulated_rotated[:,np.shape(roi_simulated_rotated)[1]//2,:]
    # plt.figure()
    # imgplot = plt.imshow(array_32, cmap='gray')

    # array_33 = roi_simulated_rotated[:,:,np.shape(roi_simulated_rotated)[2]//2]
    # plt.figure()
    # imgplot = plt.imshow(array_33, cmap='gray')

    center_slice_number = np.shape(roi_simulated_rotated)[0]//2

    roi_2d = np.transpose(roi_simulated_rotated[center_slice_number-1:center_slice_number+2,:,:], (1, 2, 0))

    processing = ScaleIntensity()

    roi_3 = processing(roi_2d)

    roi_im = im.fromarray(roi_3, mode="RGB")

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    roi_4 = preprocess(roi_im)


def plot_features():
    dataframe = pd.read_csv('C:/Users/Ran.Yan/Workspace/Project_Bone/DeepBone_Data/features_sim/features_sim_train.csv')
    dataframe = dataframe.iloc[2:12:, 60:72]
    rf_pipe = StandardScaler()
    y_pred_train = rf_pipe.fit_transform(dataframe)
    sns.heatmap(y_pred_train, cmap='bwr')


def main():
    plot_roi()



if __name__ == "__main__":
    main()



