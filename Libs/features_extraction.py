#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 15:28:30 2019

@author: valeriya
"""

import numpy as np
import scipy
from scipy import ndimage
from scipy.stats import kurtosis, skew
from mahotas.features import tas, pftas, zernike_moments
from mahotas.features.texture import haralick

from mahotas.thresholding import otsu
import cv2
from sklearn.svm import LinearSVC
from PIL import Image, ImageEnhance
import skimage
from skimage.feature import local_binary_pattern
from skimage.measure import label
from torch.utils.data import Dataset, DataLoader
import os
import time
import Libs.bandpass_segmentation as bs
from skimage import exposure, io
from tqdm import tqdm

class Preprocess:
    def __init__(self, path_for_images, max_size=None):
        
        assert isinstance(path_for_images, list), 'Please provide a list of paths for images;'\
                                                   'if there is only one path, put it into list.'
        self.path_for_images = path_for_images
        self.max_size = max_size
        
    def class_name(self):
        return self.__class__.__name__
    
    def calc_mean_var(self):
        
#         assert isinstance(self.path_for_images, list), 'Please provide a list of paths for images;'\
#                                                    'if there is only one path, put it into list.'
        allImgs = []
        
        num_folders = len(self.path_for_images)
        
        for f in range(num_folders):
            for files in os.listdir(self.path_for_images[f]):
                img = Image.open(os.path.join(self.path_for_images[f], files))
                img = np.array(img)[:,:,0]
                img = (img - img.min()) / (img - img.min()).max()
                img *= 255
                img = img.astype(np.uint8)
                allImgs = np.concatenate((allImgs, img.flatten()))
        
        return allImgs.mean(), allImgs.std()
    
    def mean_var_norm(self, img, mean, std):
        img = (img - mean) / std
        return img      
        
 
     
    def calc_padding(self):
        
#         assert isinstance(self.path_for_images, list), 'Please provide a list of paths for images;'\
#                                                    'if there is only one path, put it into list.'
        
        num_folders = len(self.path_for_images)
        
        max_shape0 = 0
        max_shape1 = 0
        
        for f in range(num_folders):
            for files in os.listdir(self.path_for_images[f]):
                img = Image.open(os.path.join(self.path_for_images[f], files))
                img = np.array(img)
                if len(img.shape)==3:
                    img = img[:,:,0]
                    
                img = (img - img.min()) / (img - img.min()).max()
                img *= 255
                img = img.astype(np.uint8)
                if img.shape[0] > max_shape0:
                    max_shape0 = img.shape[0]
                if img.shape[1] > max_shape1:
                    max_shape1 = img.shape[1] 
                    
        return max_shape0, max_shape1
    
    def padding(self, img, max_shape0, max_shape1):
        
      
        if self.max_size is None:
            self.max_size = max(max_shape0, max_shape1) + 10
        
        
        pad_size = ( ((self.max_size - img.shape[0])//2, (self.max_size - img.shape[0])//2), ((self.max_size - img.shape[1])//2, (self.max_size - img.shape[1])//2) )
        
        img_padded = np.pad(img, pad_size, 'constant')
        
        if img_padded.shape[0] < self.max_size:
            img_padded = np.pad(img_padded, ((self.max_size - img_padded.shape[0], 0), (0,0)), 'constant')
        if img_padded.shape[1] < self.max_size:
            img_padded = np.pad(img_padded, ((0, 0), (0,(self.max_size - img_padded.shape[1]))), 'constant')

        return img_padded
                               
    
class HaralickFeatures:
    
    def class_name(self):
        return self.__class__.__name__
    
    def __call__(self, image):
     # calculate haralick texture features for 4 types of adjacency
        textures = haralick(image)

       # take the mean and return it
        ht_mean = textures.mean(axis=0)
        return ht_mean
    
    
class TAS:
    def class_name(self):
        return self.__class__.__name__
    
    def __call__(self, img, thresh=None, margin=None):  
        
        if (thresh and margin) is None:
            tas_feat = pftas(img)
        else:
            assert isinstance(thresh(int, float)), 'thresh must be int of float type'

            tas_feat = tas(img, thresh, margin)
        return tas_feat
        

class ZernikeMoments:
    def __init__(self, radius):
        self.radius = radius
        
    def class_name(self):
        return self.__class__.__name__
    
    def __call__(self, image):
        radius = self.radius
        if radius==None:
            radius = max(image.shape[0], image.shape[1])//2
        if image.max() == 1:
            binary = 1 * (image > 0)
        if image.max() == 255:
            binary = 255 * (image > 0)
        else:
            binary = int(image.max()) * (image > 0)
        return zernike_moments(binary, radius)

class CenterMass:
        
    def class_name(self):
        return self.__class__.__name__
      
    def __call__(self, img):  
        return np.array(ndimage.measurements.center_of_mass(img))
    
class ChromatinFeatures:
    def __init__(self, coeff=1, thresh=0.05):
        self.coeff = coeff
        self.thresh = thresh
        
    def class_name(self):
        return self.__class__.__name__
    
    def __call__(self, img):

        img_adapteq = img

        background = skimage.morphology.area_opening(img_adapteq)
        im_bgadjusted = img_adapteq - background
        filtered = scipy.signal.medfilt(im_bgadjusted)
        segmented_chromatin = bs.bandPassSegm(filtered, 1, 3, self.coeff, self.thresh)
        small_removed  = skimage.morphology.remove_small_holes((segmented_chromatin.max() - segmented_chromatin).astype('bool'), 10)
        small_removed = np.logical_not(small_removed)
        labeled = label(small_removed)
        
        num_chromatine = len(np.unique(labeled)) - 1
        
        min_area = 1000000
        max_area = 0
        
        areas = []
        labeled_copy = labeled.copy()
        for i in np.unique(labeled)[1:]:
            labeled_copy[labeled!=i] = 0
            labeled_copy[labeled==i] = 1
            cur_area = np.sum(labeled_copy>0)
            if cur_area > max_area: max_area = cur_area
            if cur_area < min_area: min_area = cur_area  
            areas.append(cur_area)
        
        mean_area = np.array(areas).mean()
        total_area =  np.sum(np.array(areas))
        
        return min_area, max_area, mean_area, total_area        
    
def calculate_brightness(image):
    greyscale_image = image.convert('L')
    histogram = greyscale_image.histogram()
    pixels = sum(histogram)
    brightness = scale = len(histogram)

    for index in range(0, scale):
        ratio = histogram[index] / pixels
        brightness += ratio * (-scale + index)

    return 1 if brightness == 255 else brightness / scale


def extract(path_with_images, dict_feat, preprocess, objects_num, preproc, enhance, normalize): #features is dict
    print('Extraction started')
    
    if 'meanvar' in preproc:
        print('<========== Calculating standard deviation and mean ==========>')
        mean, std = preprocess.calc_mean_var()
            
    if 'pad' in preproc:
        print('<========== Calculating padding size ===========>')
        max0, max1 = preprocess.calc_padding()
        
    data = []
    
    # Create new key in dictionary with features names
    dict_feat['feature_names'] = []
    feat_idx = 0
    for n in dict_feat['features']:
        innerlist = []
        for i in range(dict_feat['feat_num'][feat_idx]):
            innerlist.append(n.class_name() + '_#%02d' % (i+1) )
        dict_feat['feature_names'].append(innerlist)
        dict_feat[n.class_name()] = np.zeros((objects_num, dict_feat['feat_num'][feat_idx]))
        feat_idx += 1
    lst = os.listdir(path_with_images)
    lst.sort()
    # Load images from path_with_images and collect all required features
    for p, files in enumerate(tqdm(lst)):
        cell = {}
        
        if enhance:   
            enhancer = ImageEnhance.Brightness(img)
            br = calculate_brightness(img)
            base_br = 0.09
            factor = base_br / br            
            
            img = enhancer.enhance(factor)
            print(calculate_brightness(img))
            
        img = cv2.imread(os.path.join(path_with_images, files), cv2.IMREAD_UNCHANGED)

        if normalize:
            img = (img - img.min()) / (img - img.min()).max()
            img *= 255
            img = img.astype(np.uint8)

        if 'meanvar' in preproc:
            img = preprocess.mean_var_norm(img, mean, std)
            img = (img - img.min()) / (img - img.min()).max()
            img *= 255
            img = img.astype(np.uint8)
        if 'pad' in preproc:
            img = preprocess.padding(img, max0, max1)

        for idx, n in enumerate(dict_feat['features']):
            dict_feat[n.class_name()][idx] = n(img)
            
            for f in range(len(dict_feat['feature_names'][idx])): 
                cell.update({str(dict_feat['feature_names'][idx][f]): dict_feat[n.class_name()][idx][f]})

        cell.update({'cell_name': files.split('.')[0]})
        data.append(cell)
    return data
    

