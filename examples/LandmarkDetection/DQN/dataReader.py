#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: dataReader.py
# Author: Amir Alansary <amiralansary@gmail.com>

import warnings
warnings.simplefilter("ignore", category=ResourceWarning)

import numpy as np
import SimpleITK as sitk
from tensorpack import logger
from IPython.core.debugger import set_trace


__all__ = ['filesListBrainMRLandmark','NiftiImage']


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

#######################################################################
## extract points from txt file
def extractPointsTXT(filename):
    x = []
    y = []
    z = []
    with open(filename) as f:
        for line in f:
            point = line.split()
            x.append(float(point[0]))
            y.append(float(point[1]))
            z.append(float(point[2]))

    return x,y,z

#######################################################################
## extract points from vtk file
def getLandmarksFromVTKFile(file):
    ''' 0-2 RV insert points
        1 -> RV lateral wall turning point
        3 -> LV lateral wall mid-point
        4 -> apex
        5-> center of the mitral valve
    '''
    with open(file) as fp:
        landmarks = []
        for i, line in enumerate(fp):
            if i == 5:
                landmarks.append([float(k) for k in line.split()])
            elif i == 6:
                landmarks.append([float(k) for k in line.split()])
            elif i > 6:
                landmarks = np.asarray(landmarks).reshape((-1,3))
                landmarks[:,[0, 1]] = -landmarks[:,[0, 1]]
                return landmarks

#######################################################################
## extract points from txt file
def getLandmarksFromTXTFile(file):
    ''' 1->3 Splenium of corpus callosum
            (outer aspect, inferior tip and inner aspect (1,2,3)),
        4,5 Genu of corpus callosum (outer and inner aspect (4,5)),
        6,7 Superior and inferior aspect of pons (6,7),
        8,16 Superior and inferior aspect cerebellum (8,16),
        9 Fourth ventricle (9),
        10->13 Putamen posterior and anterior (10,11)(left), (12,13)(right),
        14,15 Anterior and posterior commissure (14,15),
        17,18 Anterior tip of lateral ventricle (left and right) (17,18),
        19,20 Inferior tip of lateral ventricle (left and right) (19,20)
    '''
    with open(file) as fp:
        landmarks = []
        for i, line in enumerate(fp):
            landmarks.append([float(k) for k in line.split(',')])
        landmarks = np.asarray(landmarks).reshape((-1,3))
        return landmarks


###############################################################################

class filesListBrainMRLandmark(object):
    """ A class for managing train files for mri cardio data

        Attributes:
        files_list: Two or on textfiles that contain a list of all images and (landmarks)
        returnLandmarks: Return landmarks if task is train or eval (default: True)
    """
    def __init__(self, files_list=None, returnLandmarks=True):
        # check if files_list exists
        assert files_list, 'There is no directory containing files list'
        # read image filenames
        self.images_list = [line.split('\n')[0] for line in open(files_list[0].name)]
        # read landmark filenames if task is train or eval
        self.returnLandmarks = returnLandmarks
        if self.returnLandmarks:
            self.landmarks_list = self. _listLandmarks([line.split('\n')[0] for line in open(files_list[1].name)])
            assert len(self.images_list)== len(self.landmarks_list), 'number of image files is not equal to number of landmark files'


    @property
    def num_files(self):
        return len(self.images_list)

    # _listLandmarks should be removed and replaced with one landmark per file
    def _listLandmarks(self, landmark_files):
        landmarks = []
        for filename in landmark_files:
            points = getLandmarksFromTXTFile(filename)
            # landmark point 13 ac - 14 pc
            landmark = np.round(points[14]).astype('int')
            landmarks.append(landmark)
        return landmarks

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
                if self.returnLandmarks:
                    landmark = self.landmarks_list[idx]
                else:
                    landmark = None
                # extract filename from path
                image_filename = self.images_list[idx][:-7]
                yield image, landmark, image_filename, sitk_image.GetSpacing()

###############################################################################

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
            np_image = sitk.GetArrayFromImage(sitk_image)
            # threshold image between p10 and p98 then re-scale [0-255]
            p0 = np_image.min().astype('float')
            p10 = np.percentile(np_image,10)
            p99 = np.percentile(np_image,99)
            p100 = np_image.max().astype('float')
            # logger.info('p0 {} , p5 {} , p10 {} , p90 {} , p98 {} , p100 {}'.format(p0,p5,p10,p90,p98,p100))
            sitk_image = sitk.Threshold(sitk_image,
                                        lower=p10,
                                        upper=p100,
                                        outsideValue=p10)
            sitk_image = sitk.Threshold(sitk_image,
                                        lower=p0,
                                        upper=p99,
                                        outsideValue=p99)
            sitk_image = sitk.RescaleIntensity(sitk_image,
                                               outputMinimum=0,
                                               outputMaximum=255)

        # Convert from [depth, width, height] to [width, height, depth]
        image.data = sitk.GetArrayFromImage(sitk_image).transpose(2,1,0)#.astype('uint8')
        image.dims = np.shape(image.data)

        return sitk_image, image
