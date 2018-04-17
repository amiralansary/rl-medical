#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: sampleTrain.py
# Author: Amir Alansary <amiralansary@gmail.com>



import warnings
warnings.simplefilter("ignore", category=ResourceWarning)



import numpy as np
import SimpleITK as sitk
from tensorpack import logger
from IPython.core.debugger import set_trace


__all__ = ['files', 'filesCardio', 'filesFetalUS', 'filesListBrainMRPlane',
           'filesCardioPlane', 'filesListCardioMRPlane', 'NiftiImage']

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
    os.chdir(os.path.abspath(dirpath))
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
#######################################################################

class files(object):
    """ A class for managing train files

        Attributes:
        directory: input data directo
    """

    def __init__(self, directory=None):

        assert directory, 'There is no directory containing training files given'

        self.dir = directory

        # todo make it generic for directories and files with different scenarios
        self.images_list = self._listImages()
        self.landmarks_list = self._listLandmarks()
        self.all_landmarks_list = self._listLandmarks_all()


    def _listImages(self):

        childDirs = listFiles(self.dir,'*')

        image_files = []

        for child in childDirs:
            dir_path = os.path.join(self.dir, child)
            if not(os.path.isdir(dir_path)): continue
            # todo: extend to all nifti image extensions
            file_name = listFiles(dir_path,'*.nii.gz')
            file_path = os.path.join(dir_path, file_name[0])
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
            points = np.array(extractPointsXML(file_path))
            landmarks.append(np.array(points[:,2]))

        return landmarks


    def _listLandmarks_all(self):
        # extend directory path
        current_dir = self.dir + '/landmarks'
        childDirs = listFiles(current_dir,'*.txt')
        landmarks = []

        for child in childDirs:
            file_name = os.path.join(current_dir, child)
            file_path = os.path.join(current_dir, file_name)
            points = np.array(extractPointsTXT(file_path))
            landmark = np.array(points) # all landmark point
            landmarks.append(landmark)

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
                image_filename = self.images_list[idx][:-7]
                yield image, landmark, image_filename

    @property
    def num_files(self):
        return len(self.images_list)

    # def _get_target_loc(self,filename):
    #     ''' return the center of mass of a given label (target location)
    #     '''
    #     label_image = NiftiImage().decode_nifti(self.label_file)
    #     return np.round(center_of_mass(label_image.data))
###############################################################################

class filesListCardioMRPlane(files):
    """ A class for managing train files for Ozan mri cardio data

        Attributes:
        directory: input data directo
    """
    def __init__(self, directory=None, files_list=None):

        assert directory, 'There is no directory containing training files given'
        assert files_list, 'There is no directory containing files list'

        self.dir = directory
        self.files_list = [line.split('\n')[0] for line in open(files_list)]
        # todo make it generic for directories and files with different scenarios
        # self.images_3d_list = self._listImages('/3DLV/')
        self.images_3d_list = self._listImages('/3DLV_iso_1mm/')
        self.images_2ch_list = self._listImages('/2CH_rreg/')
        self.images_4ch_list = self._listImages('/4CH_rreg_fix_orientation/')
        self.landmarks_list = self._listLandmarks('/landmarks_new/')


    @property
    def num_files(self):
        return len(self.files_list)

    def _listImages(self, suffix):
        # extend directory path
        current_dir = self.dir + suffix
        image_files = []
        for filename in self.files_list:
            file_path = os.path.join(current_dir, filename) #+ '.nii.gz')
            image_files.append(file_path)

        return image_files

    def _listLandmarks(self, suffix):
        # extend directory path
        current_dir = self.dir + suffix
        landmark_files = []
        for filename in self.files_list:
            filename = filename[:-7] + '.vtk'
            file_path = os.path.join(current_dir, filename)
            landmark_files.append(file_path)

        return landmark_files


    def sample_circular(self,shuffle=False):
        """ return a random sampled ImageRecord from the list of files
        """
        if shuffle:
            indexes = rng.choice(x,len(x),replace=False)
        else:
            indexes = np.arange(self.num_files)

        while True:
            for idx in indexes:
                # print('============================================')
                # print('images_3d_list[idx] {} \nimages_2ch_list[idx] {} \nimages_4ch_list[idx] {}'.format(self.images_3d_list[idx].split('/')[-1],self.images_2ch_list[idx].split('/')[-1],self.images_4ch_list[idx].split('/')[-1]))

                sitk_image3d, _ =NiftiImage().decode(self.images_3d_list[idx])
                sitk_image2ch, _=NiftiImage().decode(self.images_2ch_list[idx])
                sitk_image4ch, _=NiftiImage().decode(self.images_4ch_list[idx])
                landmarks = getLandmarksFromVTKFile(self.landmarks_list[idx])
                filename = self.images_3d_list[idx][:-7]
                # transform landmarks to image coordinates
                landmarks = [sitk_image3d.TransformPhysicalPointToContinuousIndex(point) for point in landmarks]

                yield sitk_image3d, sitk_image2ch, sitk_image4ch, landmarks, filename



###############################################################################

class filesListBrainMRPlane(files):
    """ A class for managing train files for Ozan mri cardio data

        Attributes:
        directory: input data directo
    """
    def __init__(self, directory=None, files_list=None):

        assert directory, 'There is no directory containing training files given'
        assert files_list, 'There is no directory containing files list'

        self.dir = directory
        self.files_list = [line.split('\n')[0] for line in open(files_list)]
        # todo make it generic for directories and files with different scenarios
        # self.images_3d_list = self._listImages('/3DLV/')

        self.images_3d_list = self._listImages('/Normalized_MNI/')
        self.landmarks_list = self._listLandmarks('/LandmarksMAN/VoxelCoordinates/')


    @property
    def num_files(self):
        return len(self.files_list)

    def _listImages(self, suffix):
        # extend directory path
        current_dir = self.dir + suffix
        image_files = []
        for filename in self.files_list:
            file_path = os.path.join(current_dir, filename) #+ '.nii.gz')
            image_files.append(file_path)

        return image_files

    def _listLandmarks(self, suffix):
        # extend directory path
        current_dir = self.dir + suffix
        landmark_files = []
        for filename in self.files_list:
            filename = filename[:-32] + '.txt'
            file_path = os.path.join(current_dir, filename)
            landmark_files.append(file_path)

        return landmark_files


    def sample_circular(self,shuffle=False):
        """ return a random sampled ImageRecord from the list of files
        """
        if shuffle:
            indexes = rng.choice(x,len(x),replace=False)
        else:
            indexes = np.arange(self.num_files)

        while True:
            for idx in indexes:
                # print('============================================')
                # print('images_3d_list[idx] {} \nimages_2ch_list[idx] {} \nimages_4ch_list[idx] {}'.format(self.images_3d_list[idx].split('/')[-1],self.images_2ch_list[idx].split('/')[-1],self.images_4ch_list[idx].split('/')[-1]))

                sitk_image3d, _ =NiftiImage().decode(self.images_3d_list[idx])
                landmarks = getLandmarksFromTXTFile(self.landmarks_list[idx])
                filename = self.images_3d_list[idx][:-7]

                yield sitk_image3d, landmarks, filename










###############################################################################

class filesCardioPlane(object):
    """ A class for managing train files for Ozan mri cardio data

        Attributes:
        directory: input data directory
    """

    def __init__(self, directory=None):

        assert directory, 'There is no directory containing training files given'

        self.dir = directory

        # todo make it generic for directories and files with different scenarios
        self.images_3d_list = self._listImages('/3DLV_1mm_iso/')
        # self.images_3d_list = self._listImages('/3DLV/')
        self.images_2ch_list = self._listImages('/2CH_rreg/')
        self.images_4ch_list = self._listImages('/4CH_rreg_fix_orient/')

    def _listImages(self,suffix):
        # extend directory path
        current_dir = self.dir + suffix
        childDirs = listFiles(current_dir,'*.nii.gz')
        image_files = []

        for child in childDirs:
            file_name = os.path.join(current_dir, child)
            file_path = os.path.join(current_dir, file_name)
            image_files.append(file_path)

        return image_files


    def sample_circular(self,shuffle=False):
        """ return a random sampled ImageRecord from the list of files
        """
        if shuffle:
            indexes = rng.choice(x,len(x),replace=False)
        else:
            indexes = np.arange(self.num_files)

        while True:
            for idx in indexes:
                # print('============================================')
                # print('images_3d_list[idx] {} \nimages_2ch_list[idx] {} \nimages_4ch_list[idx] {}'.format(self.images_3d_list[idx].split('/')[-1],self.images_2ch_list[idx].split('/')[-1],self.images_4ch_list[idx].split('/')[-1]))

                sitk_image3d, _ =NiftiImage().decode(self.images_3d_list[idx])
                sitk_image2ch, _=NiftiImage().decode(self.images_2ch_list[idx])
                sitk_image4ch, _=NiftiImage().decode(self.images_4ch_list[idx])
                image_filename = self.images_3d_list[idx][:-7]

                yield sitk_image3d, sitk_image2ch, sitk_image4ch, image_filename

    @property
    def num_files(self):
        return len(self.images_3d_list)

###############################################################################
###############################################################################

class filesCardio(files):
    """ A class for managing train files for Ozan mri cardio data

        Attributes:
        directory: input data directo
    """

    def _listImages(self):
        # extend directory path
        current_dir = self.dir + '/images'
        childDirs = listFiles(current_dir,'*.nii.gz')
        image_files = []

        for child in childDirs:
            file_name = os.path.join(current_dir, child)
            file_path = os.path.join(current_dir, file_name)
            image_files.append(file_path)

        return image_files


    def _listLandmarks(self):
        # extend directory path
        current_dir = self.dir + '/landmarks'
        childDirs = listFiles(current_dir,'*.txt')
        landmarks = []

        for child in childDirs:
            file_name = os.path.join(current_dir, child)
            file_path = os.path.join(current_dir, file_name)
            points = np.array(extractPointsTXT(file_path))
            landmark = np.array(points[:,0]) # landmark point 0
            landmarks.append(landmark)

        return landmarks


    def _listLandmarks_all(self):
        # extend directory path
        current_dir = self.dir + '/landmarks'
        childDirs = listFiles(current_dir,'*.txt')
        landmarks = []

        for child in childDirs:
            file_name = os.path.join(current_dir, child)
            file_path = os.path.join(current_dir, file_name)
            points = np.array(extractPointsTXT(file_path))
            landmark = np.array(points) # all landmark point
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
                landmark = self.landmarks_list[idx]
                image_filename = self.images_list[idx][:-7]
                yield image, landmark, image_filename

###############################################################################

class filesFetalUS(files):
    """ A class for managing train files for Ozan mri cardio data

        Attributes:
        directory: input data directo
    """

    def _listImages(self):
        # extend directory path
        current_dir = self.dir + '/images'
        childDirs = listFiles(current_dir,'*.nii.gz')
        image_files = []

        for child in childDirs:
            file_name = os.path.join(current_dir, child)
            file_path = os.path.join(current_dir, file_name)
            image_files.append(file_path)

        return image_files


    def _listLandmarks(self):
        # extend directory path
        current_dir = self.dir + '/landmarks'
        childDirs = listFiles(current_dir,'*.txt')
        landmarks = []

        for child in childDirs:
            file_name = os.path.join(current_dir, child)
            file_path = os.path.join(current_dir, file_name)
            points = np.array(extractPointsTXT(file_path))
            landmark = np.array(points[:,12]) # landmark point 12
            landmarks.append(landmark)

        return landmarks

    def _listLandmarks_all(self):
        # extend directory path
        current_dir = self.dir + '/landmarks'
        childDirs = listFiles(current_dir,'*.txt')
        landmarks = []

        for child in childDirs:
            file_name = os.path.join(current_dir, child)
            file_path = os.path.join(current_dir, file_name)
            points = np.array(extractPointsTXT(file_path))
            landmark = np.array(points) # all landmark point
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
                landmark = self.landmarks_list[idx]
                image_filename = self.images_list[idx][:-7]
                yield image, landmark, image_filename


    def sample_circular_all_landmarks(self,shuffle=False):
        """ return a random sampled ImageRecord from the list of files
        """
        if shuffle:
            indexes = rng.choice(x,len(x),replace=False)
        else:
            indexes = np.arange(self.num_files)

        while True:
            for idx in indexes:
                sitk_image, image = NiftiImage().decode(self.images_list[idx])
                landmarks = self.all_landmarks_list[idx]
                image_filename = self.images_list[idx][:-7]
                yield image, landmarks, image_filename

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
        # stupid simpleitk
        image.data = sitk.GetArrayFromImage(sitk_image).transpose(2,1,0)#.astype('uint8')
        image.dims = np.shape(image.data)

        return sitk_image, image
