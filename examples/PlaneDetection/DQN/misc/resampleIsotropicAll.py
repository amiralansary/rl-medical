import SimpleITK as sitk

import  multiprocessing
num_cores = multiprocessing.cpu_count()

import os, sys
parentPath = os.path.abspath("..")
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)
from sampleTrain import trainFiles_cardio_plane


def resampleIsotropic(sitk_image, spacing_new=(1.0,1.0,1.0)):

    resampleFilter =sitk.ResampleImageFilter()
    resampleFilter.SetReferenceImage(sitk_image)
    resampleFilter.SetOutputSpacing(spacing_new)
    # (sitkBSpline, sitkNearestNeighbor, sitkBlackmanWindowedSinc,
    # sitkCosineWindowedSinc, sitkLinear, sitkGaussian,
    # sitkHammingWindowedSinc, sitkLanczosWindowedSinc,
    # sitkWelchWindowedSinc, )
    resampleFilter.SetInterpolator(sitk.sitkCosineWindowedSinc)

    # size will scale with the new spacing
    spacing_old = sitk_image.GetSpacing()
    size_old = sitk_image.GetSize()
    size_new = ((int)(size_old[0]*spacing_old[0]/spacing_new[0]),
            (int)(size_old[1]*spacing_old[1]/spacing_new[1]),
            (int)(size_old[2]*spacing_old[2]/spacing_new[2]))
    resampleFilter.SetSize(size_new)
    # resampleFilter.SetOutputOrigin(sitk_image.GetOrigin())
    resampleFilter.SetNumberOfThreads(num_cores)
    # resampleFilter.SetDefaultPixelValue((float)(image.min()))
    # sitk_image.SetDirection(sitk_image_resampled.GetDirection())
    # sitk_image.SetOrigin(sitk_image_resampled.GetOrigin())
    return resampleFilter.Execute(sitk_image)



if __name__ == "__main__":

    directory = '/vol/medic01/users/aa16914/projects/tensorpack-medical/examples/LandmarkDetection3D/ultrasound_fetal_brain_DQN/data/plane_detection/cardiac/'
    save_dir = '/vol/medic01/users/aa16914/projects/tensorpack-medical/examples/LandmarkDetection3D/ultrasound_fetal_brain_DQN/data/plane_detection/cardiac/3DLV_1mm_iso/'

    # save_dir = '/vol/medic01/users/aa16914/projects/tensorpack-medical/examples/LandmarkDetection3D/ultrasound_fetal_brain_DQN/data/plane_detection/train/'

    train_files = trainFiles_cardio_plane(directory)
    sampled_files = train_files.sample_circular()

    for i in range(train_files.num_files):

        if i<150: continue

        sitk_image, sitk_image_2ch, sitk_image_4ch, filepath = next(sampled_files)
        filename = filepath.split('/')[-1]
        # filename = filename[:-7]
        print(i, '-', filename)

        # sitk_image = resampleIsotropic(sitk_image, spacing_new=(1.0,1.0,1.0))

        # # # crop original image
        # # sitk_image_cropped = cropImage(sitk_image)
        # save_file = save_dir + filename
        # sitk.WriteImage(sitk_image, save_file)
        # # sitk.WriteImage(sitk_image_4ch, 'sitk_image_4ch.nii.gz')
        # # sitk.WriteImage(sitk_image_2ch, 'sitk_image_2ch.nii.gz')
        # # sitk.WriteImage(sitk_image_cropped, 'sitk_image_cropped.nii.gz')

