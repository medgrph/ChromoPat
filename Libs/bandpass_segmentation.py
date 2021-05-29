import numpy as np
import math
from skimage.morphology import reconstruction, square, disk, cube
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2
from PIL import Image

from skimage.feature import peak_local_max
# from skimage.morphology import watershed
from skimage.segmentation import watershed
from scipy import ndimage
from scipy.ndimage import grey_dilation, generate_binary_structure, \
        maximum_filter, minimum_filter

def meshgrid(*args):
    r""" Y,X,Z = meshgrid(np.arange(-3,4),np.arange(-2,3),np.arange(2,5))"""
    s_ind = list()
    siz = len(args)
    for t in args:
        assert (isinstance(t, np.ndarray) and t.ndim == 1), "Input arguments must be 1D ndarrays."

    for k in range(0, siz):
        s_ind.append(args[k])
        s_ind[k].shape = (1,) * k + (args[k].size,) + (1,) * (siz - 1 - k)

    for k in range(0, siz):
        for m in range(0, siz):
            if k != m:
                s_ind[k] = s_ind[k].repeat(args[m].size, axis=m)

    return s_ind


def meshgrid_(shape):
    s_ind = list()

    siz = len(shape)
    for k in range(0, siz):
        s_ind.append(np.arange(0, shape[k]))
        s_ind[k].shape = (1,) * k + (shape[k],) + (1,) * (siz - 1 - k)

    for k in range(0, siz):
        for m in range(0, siz):
            if k != m:
                s_ind[k] = s_ind[k].repeat(shape[m], axis=m)

    return s_ind

def get_gaussian_kernel(std=10, mode="gauss"):
    """Function for creation of a kernel for blurring
    Parameters
    ----------
    std : int, float
    Standard deviation of a kernel in pixels
    
    mode : str
    Selection of a kernel's mode: gaussian, asymmetric or random
    
    Returns
    -------
    
    kernel : array
    
    Kernel for blurring operation
    """

    assert (mode in ("gauss", "asymm", "random"))

    shape = (np.ceil(3*std), np.ceil(3*std))

    gaussian_kernel = np.zeros((int(shape[0]), int(shape[1])))

    if mode == "gauss":
        siz = tuple((k - 1) / 2 for k in shape)
        [y, x] = meshgrid(np.arange(-siz[0], siz[0] + 1), np.arange(-siz[1], siz[1] + 1))
        arg = -(x ** 2 + y ** 2) / (2 * std ** 2)

        h = np.exp(arg)
        eps = np.spacing(1)
#         h[h < eps * np.max(h)] = 0
        if np.sum(h) != 0:
            h = h / np.sum(h)
        kernel = h

        if mode == "asymm":
            siz = tuple((k - 1) / 2 for k in shape)

            [y, x] = meshgrid(np.arange(-siz[0], siz[0] + 1), np.arange(-siz[1], siz[1] + 1))
            arg = -(x ** 2 + y ** 2) / (2 * std ** 2)

            h = np.exp(arg)
            eps = np.spacing(1)
            h[h < eps * np.max(h)] = 0
            h[-1, :] = 1
            h[-2:, -1] = 0.5
            if np.sum(h) != 0:
                h = h / np.sum(h)
            kernel[i] = h
            std = random.randint(0, 10)

        if mode == "random":
            h = torch.rand((kernel_size, kernel_size), dtype=torch.float64)
            kernel[i] = h

    return kernel


def bandPassSegm(img, lowSigm, highSigm, coeff, thresh):
    """Segments image by bandpass filtering and thresholding.
    Parameters
    ----------
    img : 2-D array
        The input array to be segmented
    lowSigm: float
        Standard deviation of a kernel in pixels to remove noise
    highSigm : float
        Standard deviation of a kernel in pixels to highlight the background
    coeff : float
        Coefficient thatmultiplies the background. coeff = 1 corresponds to the actual bandpass filter.
    thresh : float 
        Threshold above which pixels are considered as segmented area of a sample.  
    Returns
    -------
    im_mask : array
        Binary image of segmented areas of a sample
    """
    img = img - img.min()
    img =img/ img.max()
    
    kernel_sm = get_gaussian_kernel(lowSigm)
    kernel_bg = get_gaussian_kernel(highSigm)
    
    img_sm = conv2(img, kernel_sm[::-1, ::-1], mode='same', boundary='symm')
    img_bg = conv2(img, kernel_bg[::-1, ::-1], mode='same', boundary='symm')
    
    img_sample = img_sm - img_bg * coeff
    
    im_mask = img_sample > thresh
    
    return im_mask.astype(float)

######GALA FUNCTIONS################################### 
def complement(a):
    return a.max()-a

def hminima(a, thresh):
    """Suppress all minima that are shallower than thresh.
    Parameters
    ----------
    a : array
        The input array on which to perform hminima.
    thresh : float
        Any local minima shallower than this will be flattened.
    Returns
    -------
    out : array
        A copy of the input array with shallow minima suppressed.
    """
    maxval = a.max()
    ainv = maxval - a
    return maxval - morphological_reconstruction(ainv-thresh, ainv)

def morphological_reconstruction(marker, mask, connectivity=1):
    """Perform morphological reconstruction of the marker into the mask.
    
    See the Matlab image processing toolbox documentation for details:
    http://www.mathworks.com/help/toolbox/images/f18-16264.html
    """
    sel = generate_binary_structure(marker.ndim, connectivity)
    diff = True
    while diff:
        markernew = grey_dilation(marker, footprint=sel)
        markernew = np.minimum(markernew, mask)
        diff = (markernew-marker).max() > 0
        marker = markernew
    return marker

def regional_minima(a, connectivity=1):
    """Find the regional minima in an ndarray."""
    values = np.unique(a)
    delta = (values - minimum_filter(values, footprint=np.ones(3)))[1:].min()
    marker = complement(a)
    mask = marker+delta
    return (marker == morphological_reconstruction(marker, mask, connectivity)).astype(float)
#######################################################


def remove_false_positives(img, labels):
    """Removes all segmented areas that are either too small or have too low brightness.
    Parameters
    ----------
    img : 2-D array
        The input array that is used to select to find too small and dark areas.
    labels : array
        Array corresponding to the segmented image, where
    each segmented area has its own label.
    Returns
    -------
    labels : array
        Array of segmented image cleaned from false positive segmented areas.
    """
    labels_tmp = labels.copy()
    for i in np.unique(labels):
        labels_tmp[labels!=i] = 0
        labels_tmp[labels==i] = 1
        temp = np.multiply(labels_tmp, img)
        if temp.max() < 1.5*img.mean():
            labels[labels==i] = 0
    for i in np.unique(labels)[1:]:
        x_min = np.where(labels==i)[1].min()
        x_max = np.where(labels==i)[1].max()
        y_min = np.where(labels==i)[0].min()
        y_max = np.where(labels==i)[0].max()
        area = labels[y_min:y_max, x_min:x_max]
        area = area / area.max()
        area = area[area!=0].sum()
        if area < 1000:
            labels[labels==i] = 0
    return labels

def watershedSegm(mask):
    """Watershed segmentation.
    Calculates distence transform of a binary image, 
    then supresses all minima whose depth is less then H. 
    After that computes regional minima of the H-minima transform.
    Finally, performs Watershed on the distances with the new labels.

    Parameters
    ----------
    mask : array
        The input binary segmented mask of an image.
        
    Returns
    -------
    labels : array
        Array corresponding to new segmented image, where
        each segmented area has its own label.
    """
    
    dist    = -ndimage.distance_transform_edt(mask)
    local   = hminima(dist, 2)
    local   = regional_minima(local, 8)
    markers = ndimage.label(local)[0]
    labels  = watershed(dist, markers, mask=mask)
    
    return labels

def main(img, lowSigm, highSigm, coeff, thresh, removeFalsePosit=True):
    
    if img.ndim!=2 and img.ndim==3:
        img = img[:,:,0]
#         raise Exception("'img' must be a 2-D grayscale array.")ш

    assert type(removeFalsePosit)==bool, 'removeFalsePosit must be Bool'
            
    print('Segmenting cells with Banpass filter High Std %2d, Low Std %2d' % (highSigm, lowSigm))  
    segmented_image = bandPassSegm(img, lowSigm, highSigm, coeff, thresh)
    
    print('<========== Performing Watershed segmentation ==========>')
    new_segmented_image = watershedSegm(segmented_image)
    
    if removeFalsePosit:
        print('<========== Removing false positive segmented areas ==========>')
        new_segmented_image = remove_false_positives(img, new_segmented_image)
    
    return new_segmented_image

    
    