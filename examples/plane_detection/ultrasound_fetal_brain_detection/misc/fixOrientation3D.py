import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.optimize
import functools
from math import (cos,sin,radians)

import  multiprocessing
num_cores = multiprocessing.cpu_count()

###############################################################################
def plane(x, y, params):
    ''' returns z-plane from (x,y) point with parameters (parms)
    '''
    a = params[0]
    b = params[1]
    c = params[2]
    z = a*x + b*y + c
    return z

def error(params, points):
    ''' returns the accumaletd absolute error between points (x,y,z) and
    a plane with parameters (params)
    '''
    result = 0
    for (x,y,z) in points:
        plane_z = plane(x, y, params)
        diff = abs(plane_z - z)
        result += diff**2
    return result

def projectPointOnXYPlane(points, point_in_plane):
    # get the vector and distance between each point and point_in_plane
    vector = points - point_in_plane
    dist = np.linalg.norm(vector)
    # normalize this vector and replace the z-axis with zero (standard x-y)
    unit_vector = vector/np.linalg.norm(vector)
    unit_vector[:,2] = 0
    # return new points in the direction of the vector
    return unit_vector * dist + point_in_plane


def registerLandmarks(fixedLandmarks, movingLandmarks, centerPoint):
    # initialize the transformaiton matrix
    rigid_3d_transform = sitk.VersorRigid3DTransform()
    # set identity matrix
    rigid_3d_transform.SetIdentity()
    # set the center point of rotation
    rigid_3d_transform.SetCenter([p for p in centerPoint])
    # initialize the landmark trnasformer
    fixed_points_flat = [c for p in fixedLandmarks for c in p]
    moving_points_flat = [c for p in movingLandmarks for c in p]
    landmark_transformer = sitk.LandmarkBasedTransformInitializer(
                                    transform = rigid_3d_transform,
                                    fixedLandmarks = fixed_points_flat,
                                    movingLandmarks = moving_points_flat,
                                    #referenceImage = referenceImage
                                    )
    return sitk.VersorRigid3DTransform(landmark_transformer)

def transformImage(moving_image, transformation):

    moving_array = sitk.GetArrayFromImage(moving_image)
    min = (float)(moving_array.min())
    max = (float)(moving_array.max())

    resampleFilter = sitk.ResampleImageFilter()
    resampleFilter.SetInterpolator(sitk.sitkBSpline) # sitkNearestNeighbor
    resampleFilter.SetTransform(transformation)
    resampleFilter.SetSize(moving_image.GetSize())

    origin = moving_image.GetOrigin()
    # origin = transformation.TransformPoint(origin)
    resampleFilter.SetOutputOrigin(origin)

    resampleFilter.SetOutputSpacing(moving_image.GetSpacing())
    resampleFilter.SetOutputDirection(moving_image.GetDirection())
    resampleFilter.SetDefaultPixelValue(min)
    resampleFilter.SetNumberOfThreads(num_cores)
    # resizeFilter.DebugOn()
    resampled_image = resampleFilter.Execute(moving_image)

    # fix background values within the range of [min, max]
    resampled_array = sitk.GetArrayFromImage(resampled_image)
    resampled_array[resampled_array<0.5] = 0
    resampled_array[resampled_array>255] = 255
    resampled_image_final = sitk.GetImageFromArray(resampled_array)
    resampled_image_final.CopyInformation(resampled_image)

    return resampled_image_final

def getSurfaceFromPoints(xx, yy, points):
    # optimization function - the error between gt_points and fitted plane
    opt_func = functools.partial(error, points=points)
    # initialize plane equation parameters z = a*x + b*y + c with zeros
    plane_initial_params = [0, 0, 0]
    # final parameter result values
    res = scipy.optimize.minimize(opt_func, plane_initial_params)
    plane_final_params = res.x
    # pick the origin point of the new plane at (0,0,c) equivalant to (0,0,1)
    return plane(xx, yy, plane_final_params)



def fix_orientation_3d(sitk_image, landmarks, viz=False):

    image = sitk.GetArrayFromImage(sitk_image)
    image_spacing = sitk_image.GetSpacing()
    image_dims = image.shape

    points_fixed = landmarks

    ###########################################################################
    ########################## fit landmarks plane ############################
    ###########################################################################
    # optimization function - the error between gt_points and fitted plane
    opt_func = functools.partial(error, points=points_fixed)
    # initialize plane equation parameters z = a*x + b*y + c with zeros
    plane_initial_params = [0, 0, 0]
    # final parameter result values
    res = scipy.optimize.minimize(opt_func, plane_initial_params)

    plane_final_params = res.x

    # extract new landmarks from the same x-y but on the new plane
    points_fixed[:,2] = plane(points_fixed[:,0], points_fixed[:,1], plane_final_params)

    # extract the correspondant points on the standard x-y plane containing csp point
    csp_point = points_fixed[5]
    points_moving = projectPointOnXYPlane(points_fixed, csp_point)

    ###########################################################################
    ############################### plot points ###############################
    ###########################################################################
    if viz:
        xx, yy = np.meshgrid([0,image_dims[0]], [0,image_dims[1]])
        # unpack points
        xs_fixed, ys_fixed, zs_fixed = zip(*points_fixed)
        xs_moving, ys_moving, zs_moving = zip(*points_moving)
        # prepare figure
        fig = plt.figure();
        ax = fig.add_subplot(111, projection='3d');
        # plot scatter points
        ax.scatter(xs_fixed, ys_fixed, zs_fixed, color='blue');
        ax.scatter(xs_moving, ys_moving, zs_moving, color='red');

        # plot surface
        zz_fixed = getSurfaceFromPoints(xx, yy, points_fixed)
        zz_moving = getSurfaceFromPoints(xx, yy, points_moving)
        ax.plot_surface(xx, yy, zz_fixed, alpha=0.5, color='blue');
        ax.plot_surface(xx, yy, zz_moving, alpha=0.5, color='red');

    ###########################################################################
    ###################### register landmark point sets #######################
    ###########################################################################
    points_physical_fixed = [sitk_image.TransformContinuousIndexToPhysicalPoint(p) for p in points_fixed]
    points_physical_moving = [sitk_image.TransformContinuousIndexToPhysicalPoint(p) for p in points_moving]

    transformation = registerLandmarks(fixedLandmarks= points_physical_fixed,
                                       movingLandmarks=points_physical_moving,
                                       centerPoint=points_physical_fixed[5])

    sitk_image_transformed = transformImage(sitk_image, transformation)

    # print(sitk_image.GetDirection())
    # print(sitk_image_transformed.GetDirection())

    # sitk.WriteImage(sitk_image, 'image.nii.gz')
    # sitk.WriteImage(sitk_image_transformed, 'image_transformed.nii.gz')

    ###########################################################################
    ##################### plot slice containing landmarks #####################
    ###########################################################################

    points_trans = [sitk_image.TransformPhysicalPointToContinuousIndex(transformation.TransformPoint(p)) for p in points_physical_fixed]

    if viz:
        xs_trans, ys_trans, zs_trans = zip(*points_trans)
        ax.scatter(xs_trans, ys_trans, zs_trans, color='green');

        ax.set_xlim(0,image_dims[0]);
        ax.set_ylim(0,image_dims[1]);
        ax.set_zlim(0,image_dims[2]);

        mng = plt.get_current_fig_manager();
        mng.resize(*mng.window.maxsize());

        plt.show()

        # plot slice containing landmarks before and after
        image = sitk.GetArrayFromImage(sitk_image)
        image_slice = image[(int)(csp_point[2]),:,:]
        image_transformed = sitk.GetArrayFromImage(sitk_image_transformed)
        image_transformed_slice = image_transformed[(int)(zs_trans[0]),:,:]

        fig = plt.figure();

        plt.subplot(1, 2, 1);
        plt.axis('off');
        plt.imshow(image_slice, cmap='gray');
        plt.scatter(xs_fixed, ys_fixed, color='red');

        plt.subplot(1, 2, 2);
        plt.axis('off');
        plt.imshow(image_transformed_slice, cmap='gray');
        plt.scatter(xs_trans, ys_trans, color='green');
        mng = plt.get_current_fig_manager();
        mng.resize(*mng.window.maxsize());
        plt.show();

    return sitk_image_transformed, points_trans, transformation



###############################################################################
###############################################################################

from sampleTrain import trainFiles_fetal_US, NiftiImage
import SimpleITK as sitk


if __name__ == "__main__":

    directory = '/vol/medic01/users/aa16914/projects/tensorpack-medical/examples/LandmarkDetection3D/ultrasound_fetal_brain_DQN/data/train_align/'

    save_dir = '/vol/medic01/users/aa16914/projects/tensorpack-medical/examples/LandmarkDetection3D/ultrasound_fetal_brain_DQN/data/plane_detection/train/'

    train_files = trainFiles_fetal_US(directory)
    sampled_files = train_files.sample_circular_all_landmarks()

    for i in range(len(train_files.images_list)):

        img, landmarks, filepath = next(sampled_files)
        filename = filepath.split('/')[-1]
        filename = filename[:-7]
        print(filename)

        sitk_image, image = NiftiImage().decode(filename=filepath)
        # select landmarks
        # first line
        # (0. topHead, 1. centerHead, 2. frontHead, 3. backHead, 12. CSP, 13. midpointCSP)
        # p0 = landmarks[0]
        p1 = landmarks[:,1]
        p2 = landmarks[:,2]
        p3 = landmarks[:,3]
        p12 = landmarks[:,12]
        p13 = landmarks[:,13]
        points1 = np.array((p1,p2,p3,p12,p13))

        # second line
        # (1. centerHead, 4. rightHead, 5. leftHead)
        p4 = landmarks[:,4]
        p5 = landmarks[:,5]
        points2 = np.array((p1,p4,p5))

        selected_landmarks = np.array((p1,p2,p3,p4,p5,p12,p13))

        sitk_image_transformed, points_trans, transformation = fix_orientation_3d(sitk_image,selected_landmarks, viz=True)

        sitk_image_transformed = sitk.RescaleIntensity(sitk_image_transformed,
                                                       outputMinimum=0.0,
                                                       outputMaximum=1.0)

        sitk.WriteImage(sitk_image_transformed, save_dir + 'images/' + filename + '.nii.gz')

        np.savetxt(save_dir + 'landmarks/' + filename + '.txt', points_trans, fmt='%1.4f')










###############################################################################
######################## find plane directional angles ########################
###############################################################################

# # pick the origin point of the new plane at (0,0,c) equivalant to (0,0,1)
# origin_point_gt = np.array([0.0, 0.0, plane(0,0,plane_final_params)])

# # find the normal vector using the cross product between two unit vectors of
# # in the direction of x and y of the new plane
# v1 = np.array([1,0,plane(1,0,plane_final_params)]) - origin_point_gt
# v2 = np.array([0,1,plane(0,1,plane_final_params)]) - origin_point_gt

# v_norm_fit_plane = np.array(np.cross(v1, v2))
# # print('v_norm_fit_plane {}'.format(v_norm_fit_plane))
# # normalise norm vector on the fitted plane
# v_norm_fit_plane = v_norm_fit_plane * 1.0 / np.linalg.norm(v_norm_fit_plane)
# # print('v_norm_fit_plane {}'.format(v_norm_fit_plane))

# # find the dot product between the norm and origin point sitting on the
# # new plane d = -(ax + by + cz)
# d_gt = -origin_point_gt.dot(v_norm_fit_plane)

# # Point-normal form and general form of the equation of a plane
# # ax + by + cz + d = 0 , where (a,b,c) is the normal vector
# # zz_gt = (-v_norm_fit_plane[0] * xx - v_norm_fit_plane[1] * yy - d_gt) * 1. /v_norm_fit_plane[2]
# # print('zz_gt {}'.format(zz_gt))
# zz_gt = plane(xx, yy, plane_final_params)
# # print('zz_gt {}'.format(zz_gt))

# # plot the new plane surface
# ax.plot_surface(xx, yy, zz_gt, alpha=0.5, color=[0,1,0])

# vlength = 100*np.linalg.norm(v_norm_fit_plane)
# plt.quiver(0,0,origin_point_gt[2],
#            v_norm_fit_plane[0], v_norm_fit_plane[1], v_norm_fit_plane[2],
#            pivot='tail',length=vlength,arrow_length_ratio=0.2/vlength,
#            color='black')
# plt.quiver(0,0,origin_point_gt[2],
#            v1[0],v1[1],v1[2],
#            # 1,0,a_final,
#            pivot='tail',length=vlength,arrow_length_ratio=0.2/vlength,
#            color='blue')
# plt.quiver(0,0,origin_point_gt[2],
#            v2[0],v2[1],v2[2],
#            # 0,1,b_final,
#            pivot='tail',length=vlength,arrow_length_ratio=0.2/vlength,
#            color='red')
# plt.quiver(0,0,origin_point_gt[2],
#            0,0,1,
#            pivot='tail',length=vlength,arrow_length_ratio=0.2/vlength,
#            color='green')


# ###############################################################################
# ############################## find cross lines ###############################
# ###############################################################################

# def fit_line(data):
#     # Calculate the mean of the points, i.e. the 'center' of the cloud
#     datamean = data.mean(axis=0)

#     # Do an SVD on the mean-centered data.
#     uu, dd, vv = np.linalg.svd(data - datamean)

#     # Now vv[0] contains the first principal component, i.e. the direction
#     # vector of the 'best fit' line in the least squares sense.

#     # Now generate some points along this best fit line, for plotting.

#     # I use -7, 7 since the spread of the data is roughly 14
#     # and we want it to have mean 0 (like the points we did
#     # the svd on). Also, it's a straight line, so we only need 2 points.
#     linepts = vv[0] * np.mgrid[-100:100:2j][:, np.newaxis]

#     # shift by the mean to get the line in the right place
#     linepts += datamean

#     return linepts


# # fit line to points
# linepts1 = fit_line(points1)
# linepts2 = fit_line(points2)
# ax.plot(*linepts1.T)
# ax.plot(*linepts2.T)



# ###############################################################################
# ###############################################################################
# ax.set_xlim(0,image_dims[0])
# ax.set_ylim(0,image_dims[1])
# ax.set_zlim(0,image_dims[2])

# plt.show()


###############################################################################

## landmark coding
# 0. topHead
# 1. centerHead
# 2. frontHead
# 3. backHead
# 4. rightHead
# 5. leftHead
# 6. rightVentInner
# 7. rightVentOuter
# 8. leftVentInner
# 9. leftVentOuter
# 10. rightCerebellar
# 11. leftCerebellar
# 12. CSP
# 13. midpointCSP
# 14. rightEye
# 15. leftEye


# first line
# (0. topHead, 1. centerHead, 2. frontHead, 3. backHead, 12. CSP, 13. midpointCSP)

# second line
# (1. centerHead, 4. rightHead, 5. leftHead)


# TV plane: 1,2,3,4,5,6,7,8,9,12,13
# TC plane: 10,11,12
