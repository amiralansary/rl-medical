import numpy as np
import SimpleITK as sitk

import  multiprocessing
num_cores = multiprocessing.cpu_count()


def registerLandmarks(fixedLandmarks, movingLandmarks,
                      centerPoint = None,
                      movingImage = None,
                      )

    # initialize the transformaiton matrix
    rigid_3d_transform = sitk.VersorRigid3DTransform()
    # set identity matrix
    rigid_3d_transform.SetIdentity()
    # set the center point of rotation
    if centerPoint:
        rigid_3d_transform.SetCenter(centerPoint)

    # initialize the landmark trnasformer

    landmark_transformer = sitk.LandmarkBasedTransformInitializer(
                                        Transform = rigid_3d_transform,
                                        fixedLandmarks = fixedLandmarks,
                                        movingLandmarks = movingLandmarks,
                                        #referenceImage = referenceImage
                                        )

    final_transformation = sitk.VersorRigid3DTransform(landmark_transformer)

    if movingImage:
        movingArray = sitk.GetArrayFromImage(movingImage)
        resampleFilter = sitk.ResampleImageFilter()
        resampleFilter.SetInput(movingImage)
        resampleFilter.SetTransform(final_transformation)
        # resampleFilter.SetSize(movingImage.GetSize())
        # resampleFilter.SetOutputOrigin(movingImage.GetOrigin())
        # resampleFilter.SetOutputSpacing(movingImage.GetSpacing())
        # resampleFilter.SetOutputDirection(movingImage.GetDirection())
        resampleFilter.SetDefaultPixelValue(movingArray.min())
        resampleFilter.SetNumberOfThreads(num_cores)
        # resizeFilter.DebugOn()
        img_resampled = resampleFilter.Execute(movingImage)

        return img_resampled



# #fixed_image_points, moving_image_points = point_acquisition_interface.get_points()
# fixed_image_points = [(156.48434676356158, 201.92274575468412, 68.0),
#                       (194.25413436597393, 98.55771047484492, 32.0),
#                       (128.94523819661913, 96.18284152323203, 32.0)]
# moving_image_points = [(141.46826904042848, 156.97653126727528, 48.0),
#                        (113.70102381552435, 251.76553994455645, 8.0),
#                        (180.69457220262115, 251.76553994455645, 8.0)]


# fixed_image_points_flat = [c for p in fixed_image_points for c in p]
# moving_image_points_flat = [c for p in moving_image_points for c in p]
# manual_localized_transformation = sitk.VersorRigid3DTransform(sitk.LandmarkBasedTransformInitializer(sitk.VersorRigid3DTransform(),
#                                       fixed_image_points_flat,
#                                       moving_image_points_flat))

# manual_errors_mean, manual_errors_std, manual_errors_min, manual_errors_max,_ = \
#     ru.registration_errors(manual_localized_transformation,
#                            fixed_image_points,
#                            moving_image_points,
#                            display_errors=True)
# print('After registration (manual point localization), errors (TRE) in millimeters, mean(std): {:.2f}({:.2f}), max: {:.2f}'.format(manual_errors_mean, manual_errors_std, manual_errors_max))
