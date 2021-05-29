import os
import math
from itertools import cycle

import numpy as np
import cv2
import skimage
from PIL import Image
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd

from scipy.stats.mstats import winsorize

import numpy as np
import cv2

import os
import math
from itertools import cycle


class Dataset:

    def __init__(self, image_path, file_name, channels, mask_channel, result_path, image_series=False):
        self.image_path = image_path
        self.channels = channels
        self.mask_channel = mask_channel
        self.result_path = result_path
        self.image_series = image_series
        self.file_name = file_name

    def read_series(self, image_path):
        """Read all images from path and save it as array
        param path: directory with files, str
        return img_series: array with images, array
        return img_names: list of file names, list
        """
        all_img = []
        img_names = []
        for img_name in sorted(os.listdir(image_path), key=lambda x: int(x.split('.')[0])):
            img = cv2.imread(os.path.join(image_path, img_name), cv2.IMREAD_UNCHANGED)
            img = cv2.convertScaleAbs(img)
            all_img.append(img)
            img_names.append(img_name)
        img_series = np.array(all_img)
        return img_names, img_series

    def split_image_series(self, image_path, file_name, result_path, channels):
        """
        Split image series to channels folders
        param file_dir: folder with image, str
        param file_name: full file name, str
        param res_dir: directory for result, str
        param channels: list of channels, list
        """
        print(os.path.join(image_path, file_name))
        img_series = cv2.imreadmulti(os.path.join(image_path, file_name), flags=cv2.IMREAD_UNCHANGED)
        img_series = np.array(img_series[1])
        for c in channels:
            res_dir_full = os.path.join(result_path, c)
            print(res_dir_full)
            os.makedirs(res_dir_full, exist_ok=True)
        channels = cycle(channels)
        for ind, img in enumerate(img_series):
            print(f'{os.path.join(result_path, next(channels))}/{str(ind)}.tif')
            if not cv2.imwrite(f'{os.path.join(result_path, next(channels))}/{str(ind)}.tif', img):
                raise Exception('Cannot save image')

    def split_masks_by_channels(self, masks_dir, result_path, channels):
        """
        Split image series to channels folders
        param masks_dir: folder with masks, str
        param result_path: directory for result, str
        param channels: list of channels, list
        """
        mask_names, masks_series = self.read_series(masks_dir)
        for c in channels:
            res_dir_full = os.path.join(result_path, c)
            os.makedirs(res_dir_full, exist_ok=True)
        channels = cycle(channels)
        for ind, img in enumerate(masks_series):
            if not cv2.imwrite(f'{os.path.join(result_path, next(channels))}/{str(ind)}.tif', img):
                raise Exception('Cannot save mask')

    def read_and_normalize(self, image_path):
        """
        Read image series and normalize it
        param files_dir: files directory, str
        return img_series: image array, array
        return img_names: list of file names, list
        """
        img_names, img_series = self.read_series(image_path)
        norm_img = np.zeros(img_series.shape)
        img_series = cv2.normalize(img_series, norm_img, 0, 255, cv2.NORM_MINMAX)
        return img_names, img_series

    def crop_images_by_channel(self, img_series, masks_dir, channel):
        """
        Crop image series by mask from one channel
        param img_series: array of images from one channel, array
        param masks_dir: directory with all masks splitted by channel, str
        param channel: channel name, str
        return img_series: array of cropped images, array
        """
        masks_dir = os.path.join(masks_dir, channel)
        all_masks = []
        for mask_name in sorted(os.listdir(masks_dir), key=lambda x: int(x.split('.')[0])):
            mask = cv2.imread(os.path.join(masks_dir, mask_name), cv2.IMREAD_UNCHANGED)
            all_masks.append(mask)
        masks_series = np.array(all_masks)
        cropped_img = []
        for i, msk in enumerate(masks_series):
            ret, mask = cv2.threshold(msk, 1, 255, cv2.THRESH_BINARY)
            img_bg = cv2.bitwise_and(img_series[i], img_series[i], mask=mask)
            cropped_img.append(img_bg)
        img_series = np.array(cropped_img)
        return img_series
