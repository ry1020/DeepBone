import numpy as np
from scipy import ndimage


def applyMTF(image, MTF):
    # Apply blur to image
    
    # assumes MTF axes is consistent with image
    assert image.shape == MTF.shape
    
    imageFFT = np.fft.fftn(image,axes=tuple(range(image.ndim)))
    image_filtered = abs(np.fft.ifftn(imageFFT * MTF, axes=tuple(range(image.ndim))))
    
    return image_filtered

def applySampling(image, voxSize, voxel_size_simulated):
    dimsNew = (image.shape * np.array(voxSize) // np.array(voxel_size_simulated)).astype(int)
    shape = image.shape
    x,y,z = np.meshgrid(np.linspace(0,shape[0],dimsNew[0]).astype(int),
                                 np.linspace(0,shape[1],dimsNew[1]).astype(int),
                                 np.linspace(0,shape[2],dimsNew[2]).astype(int))

    imageNew = ndimage.map_coordinates(image, [x.flatten(), y.flatten(), z.flatten()])
    imageNew = np.reshape(imageNew,dimsNew)
    
    return imageNew

def getFreqs(shape,voxSize):
    freqs = []
    for ind, Nvoxels in enumerate(shape):
        freqs.append(np.fft.fftfreq(Nvoxels,voxSize[ind]))
    return freqs

def Gaussian3DUncorrelated(xxx,yyy,zzz,std):
    return np.exp((-xxx**2/std[0]**2-yyy**2/std[1]**2-zzz**2/std[2]**2)/2)

def make3DMTFGaussian(freqs, std):
    xxx,yyy,zzz = np.meshgrid(*freqs)
    return Gaussian3DUncorrelated(xxx,yyy,zzz,std)


def NPS2noise(NPS,seed=None):
    # generates noise from a white noise and power spectra
    # note: frequency axes must match image
    # see https://www.mathworks.com/matlabcentral/fileexchange/36462-noise-power-spectrum
    
    rng = np.random.default_rng(seed)
    v = rng.random(NPS.shape,dtype=np.float64)
    F = NPS * (np.cos(2*np.pi*v) + 1j*np.sin(2*np.pi*v))
    f = np.fft.ifftn(F, axes=tuple(range(NPS.ndim)))
    noise = np.real(f) + np.imag(f)
    
    return noise

def makeNPSRamp(freqs,alpha=1):
    xxx,yyy,zzz = np.meshgrid(*freqs)
    cone = alpha * np.sqrt(xxx**2+yyy**2)
    return cone

def integrateNPS(NPS,freqs):
    df = 1
    for ind, f in enumerate(freqs):
        df = df * np.abs(f[1]-f[0]) # integrate noise power
    return np.sum(NPS) * df

def simulateImage(roi, noise_scales, resolution_scales, voxel_size_simulated, seed=None):

    VOX_SIZE = (0.05, 0.05, 0.05) # mm
    BONE_HU = 1800

    noise_std = 180 * noise_scales
    stdMTF = np.array([1.8,1.8,0.5]) * resolution_scales

    roi = (roi - np.min(roi)) / (np.max(roi) - np.min(roi)) # normalize to 0 and 1
    roi = roi * BONE_HU  # scale to bone HU

    roi = applySampling(roi, VOX_SIZE, voxel_size_simulated)

    # get frequency axis
    freqs = getFreqs(roi.shape,voxel_size_simulated)
    
    T = make3DMTFGaussian(freqs,stdMTF)
    roiT = applyMTF(roi,T)
    
    ramp = makeNPSRamp(freqs)
    
    S = (T**2)*ramp
    S = S / integrateNPS(S,freqs) * noise_std # TODO: should be squqred
    
    noiseS = NPS2noise(S,seed=seed) * (noise_std**2)
    
    roi_final = roiT+noiseS
    
    return roi_final