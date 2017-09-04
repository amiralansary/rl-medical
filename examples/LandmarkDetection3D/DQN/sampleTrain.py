#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: sampleTrain.py
# Author: Amir Alansary <amiralansary@gmail.com>

import numpy as np
import SimpleITK as sitk
from tensorpack import logger


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
## extract points from xml file
import xml.etree.ElementTree as ET

def extractPointsXML(filename):

    tree = ET.parse(filename)
    root = tree.getroot()

    x = []
    y = []
    z = []

    for point in root[1].findall('time_series/point'):
        x.append(float(point[2].text))
        y.append(float(point[3].text))
        z.append(float(point[4].text))

    return x,y,z


__all__ = ['trainFiles', 'NiftiImage']

#######################################################################

class trainFiles(object):
    """ A class for managing train files

        Attributes:
        data:               input data
        model:              model to train
        sess (tf.Session):  the current session in use.
        epoch_num (int):    number of epochs that have finished.
        local_step (int):   number of steps that have finished in the current epoch.
        global_step (int):  number of steps that have finished.
    """

    def __init__(self, directory=None):

        assert directory, 'There is no directory containing training files given'

        self.dir = directory

        # todo make it generic for directories and files with different scenarios
        self.images_list = self._listImages()
        self.landmarks_list = self._listLandmarks()


    def _listImages(self):

        childDirs = listFiles(self.dir,'*')

        image_files = []

        for child in childDirs:
            dir_path = os.path.join(self.dir, child)
            if not(os.path.isdir(dir_path)): continue
            # todo: extend to all nifti image extensions
            file_name = listFiles(dir_path,'*.nii.gz')
            file_path = os.path.join(dir_path, file_name[0])
            # logger.info(file_path)
            image_files.append(file_path)

        return image_files


    def _listLandmarks(self):

        childDirs = listFiles(self.dir,'*')
        landmarks = []

        for child in childDirs:
            dir_path = os.path.join(self.dir, child)
            if not(os.path.isdir(dir_path)): continue

            file_name = listFiles(dir_path,'*.mps')
            file_path = os.path.join(dir_path, file_name[0])
            # logger.info(file_path)
            points = np.array(extractPointsXML(file_path))
            landmarks.append(np.array(points[:,2]))

        return landmarks

    def sample_random(self):
        """ return a random sampled ImageRecord from the list of files
        """
        # todo: fix seed for a fair comparison between models
        random_idx = np.random.randint(low=0, high=len(self.images_list))
        sitk_image, image = NiftiImage().decode(self.images_list[random_idx])
        landmark = np.array(sitk_image.TransformPhysicalPointToIndex(self.landmarks_list[random_idx]))
        return image, landmark, random_idx


    def sample_circular(self,shuffle=False):
        """ return a random sampled ImageRecord from the list of files
        """
        if shuffle:
            indexes = rng.choice(x,len(x),replace=False)
        else:
            indexes = np.arange(self.num_files)

        while True:
            for idx in indexes:
                sitk_image, image = NiftiImage().decode(self.images_list[idx])
                landmark = np.array(sitk_image.TransformPhysicalPointToIndex(self.landmarks_list[idx]))
                # logger.info('image{} {}'.format(idx, self.images_list[idx]))
                image_filename = self.images_list[idx]
                yield image, landmark, image_filename


    @property
    def num_files(self):
        return len(self.images_list)

    # def _get_target_loc(self,filename):
    #     ''' return the center of mass of a given label (target location)
    #     '''
    #     label_image = NiftiImage().decode_nifti(self.label_file)
    #     return np.round(center_of_mass(label_image.data))

# ===================================================================
# ====================== Nifti Image Class ==========================
# ===================================================================
class ImageRecord(object):
  '''image object to contain height,width, depth and name '''
  pass


class NiftiImage(object):
    """Helper class that provides TensorFlow image coding utilities."""
    def __init__(self):
        pass

    def _is_nifti(self,filename):
        """Determine if a file contains a nifti format image.
        Args
          filename: string, path of the image file
        Returns
          boolean indicating if the image is a nifti
        """
        extensions = ['.nii','.nii.gz','.img','.hdr']
        return any(i in filename for i in extensions)

    def decode(self, filename,label=False):
        """ decode a single nifti image
        Args
          filename: string for input images
          label: True if nifti image is label
        Returns
          image: an image container with attributes; name, data, dims
        """
        image = ImageRecord()
        image.name = filename
        assert self._is_nifti(image.name), "unknown image format for %r" % image.name

        if label:
          sitk_image = sitk.ReadImage(image.name, sitk.sitkInt8)
        else:
          sitk_image = sitk.ReadImage(image.name, sitk.sitkFloat32)
        # get dimensions
        # image.dims = (sitk_image.GetWidth(), sitk_image.GetHeight(), sitk_image.GetDepth())
        # print(image.dims)
        # image.height  = sitk_image.GetHeight()
        # image.width   = sitk_image.GetWidth()
        # image.depth   = sitk_image.GetDepth()
        # Convert from [depth, width, height] to [width, height, depth]
        image.data = sitk.GetArrayFromImage(sitk_image)#.transpose(2,1,0)
        image.dims = np.shape(image.data)
        # print('target val before normalization = ', image.data[65,52,63])
        # print(image.data.shape)

        if not label:
            # todo: imporved normalization using percentiles
          image.data = image.data * (255.0 / image.data.max())
          # Convert from [0, 255] -> [-0.5, 0.5] floats.
          # image.data = image.data * (1. / image.data.max()) - 0.5

        return sitk_image, image
