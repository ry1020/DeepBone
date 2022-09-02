
import numpy as np
import functools
from numpy.linalg import norm, svd
import math

def angle_between_vectors(v1, v2 = np.array([0,0,1])):

    radian = np.arccos(np.absolute(np.dot(v1,v2)) / (norm(v1)*norm(v2)) )  #Compute only angles smaller than 90 degrees
    angle = radian/math.pi*180

    return angle

def GradientStructureTensor3D(I):
    """
    Gradient structure tensor
    """

    Iag = np.gradient(I)
    Gv = np.array([I.flatten() for I in Iag])

    dyadicGv = np.array([np.outer(Gv[:,ind],Gv[:,ind]) for ind in range(Gv.shape[1])])
    gst = np.sum(dyadicGv,axis=0)

    return gst

def compute_structure_dir(volume):
    gst = GradientStructureTensor3D(volume)

    U, S, Vh = svd(gst) # returns a matrix of eigenvectors U and corresponding eigenvalues S
    angle = np.array([angle_between_vectors(U[:,0]), angle_between_vectors(U[:,1]), angle_between_vectors(U[:,2])])

    return np.concatenate((angle, S))

def compute_structure_dir_parallel(volumeList, numWorkers=None):
    from multiprocessing import cpu_count, Pool
    
    if numWorkers is None:
        numWorkers = cpu_count() - 2

    with Pool(numWorkers) as pool:
        featureVectors = pool.map(functools.partial(compute_structure_dir),volumeList)
    
    featureVectorArray = np.vstack(featureVectors)

    return featureVectorArray


def compute_mil_dir(img_strengths, idx):
    min_dir_eigenvector = img_strengths['m00','m01','m02'].iloc[idx, :].to_numpy()
    inter_dir_eigenvector = img_strengths['m10','m11','m12'].iloc[idx, :].to_numpy()
    main_dir_eigenvector = img_strengths['m20','m21','m22'].iloc[idx, :].to_numpy()
    mil_eigenvalues = img_strengths['D1','D2','D3'].iloc[idx, :].to_numpy()
    min_angle = angle_between_vectors(min_dir_eigenvector)
    inter_angle = angle_between_vectors(inter_dir_eigenvector)
    main_angle = angle_between_vectors(main_dir_eigenvector)

    mil_dir_features = np.concatenate((np.array([min_angle,inter_angle,main_angle]),mil_eigenvalues), -1)
    mil_dir_features = np.expand_dims(mil_dir_features, axis = 0)

    return mil_dir_features