
import numpy as np
import functools
from numpy.linalg import norm, svd
import math

def angle_between_vectors(v1, v2 = np.array([0,0,1])):

    radian = np.arccos(np.absolute(np.dot(v1,v2)) / (norm(v1)*norm(v2)) )  #Compute only angles smaller than 180 degrees
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
    min_dir_eigenvector = img_strengths.iloc[idx, 8:11].to_numpy()
    inter_dir_eigenvector = img_strengths.iloc[idx, 11:14].to_numpy()
    main_dir_eigenvector = img_strengths.iloc[idx, 14:17].to_numpy()
    mil_eigenvalues = img_strengths.iloc[idx, 17:20].to_numpy()
    min_angle = angle_between_vectors(min_dir_eigenvector)
    inter_angle = angle_between_vectors(inter_dir_eigenvector)
    main_angle = angle_between_vectors(main_dir_eigenvector)

    mil_dir_features = np.concatenate((np.array([min_angle,inter_angle,main_angle]),mil_eigenvalues), -1)
    mil_dir_features = np.expand_dims(mil_dir_features, axis = 0)

    return mil_dir_features