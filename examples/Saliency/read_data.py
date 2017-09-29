#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: sampleTrain.py
# Author: Amir Alansary <amiralansary@gmail.com>

import cv2
import numpy as np
import pandas as pd
# import SimpleITK as sitk

from tensorpack import logger
from tensorpack import RNGDataFlow

#######################################################################
## list file/directory names
import glob
import os
import re

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def tryint(s):
    try:
        return int(s)
    except:
        return s

def listFiles(dirpath, dirnames):
    curpath = os.getcwd()
    os.chdir(dirpath)
    f = glob.glob(dirnames)
    f.sort(key=alphanum_key)
    os.chdir(curpath)
    return f


#######################################################################

class CamCAN(RNGDataFlow):
    """
    Provide methods to access metadata for CamCAN dataset.
    """

    def __init__(self, dir=None,shuffle=None):
        self.dir = dir
        self.shuffle = shuffle
        self.imglist = self.get_image_list() # list contains (img_file, label)

    def get_image_list(self):
        """
        Returns:
            list: list of (image filename, label)
        """

        # todo make it generic for directories and files with different scenarios
        self.images_list = self._listImages()
        self.labels_list = self._listLabels()

        return [(image, label) for image, label in zip(self.images_list, self.labels_list)]

    def _listImages(self):

        imgs_dir = self.dir + '/slices'
        img_files = listFiles(imgs_dir,'*')
        img_files = [os.path.join(imgs_dir,img_file) for img_file in img_files]
        # logger.info('img_files = {}'.format(img_files))
        return img_files


    def _listLabels(self):

        labels_file = self.dir + '/clean_participant_data.csv'
        df = pd.read_csv(labels_file)
        gender_code = df.gender_code.tolist()
        # logger.info('gender_code {}'.format(gender_code))
        return gender_code

    def get_data(self):
        idxs = np.arange(len(self.imglist))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            fname, label = self.imglist[k]
            im = cv2.imread(fname, cv2.IMREAD_COLOR)
            yield [im, label]

    def size(self):
        return len(self.imglist)

