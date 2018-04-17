import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import ndimage

import SimpleITK as sitk
import numpy as np
from copy import deepcopy


import scipy.optimize
from functools import partial

import time
from collections import namedtuple
Plane = namedtuple('Plane', ['grid', 'norm', 'origin', 'params', 'points'])

###############################################################################
import  multiprocessing
num_cores = multiprocessing.cpu_count()


###############################################################################
import os, sys
parentPath = os.path.abspath("..")
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)
from sampleTrain import trainFiles_cardio_plane

from IPython.core.debugger import set_trace
# set_trace()

###############################################################################
################################ fit Plane Points #############################
###############################################################################
def cart2sph(x, y, z):
    rho = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arccos(z / rho) # np.arctan2(np.sqrt(y**2, x**2), z)
    theta = np.arctan2(y, x)
    return(rho, theta, phi)

def sph2cart(rho, theta, phi):
    x = rho * np.sin(theta) * np.cos(phi)
    y = rho * np.sin(theta) * np.sin(phi)
    z = rho * np.cos(theta)
    return (x, y, z)


def cartPlane(x, y, params):
    ''' returns z-plane from (x,y) point and cartesian parameters (parms)
        z = a*x + b*y + c
    '''
    a = params[0]
    b = params[1]
    c = params[2]

    z = a*x + b*y + c
    return z

def normPlaneFromCartPlane(plane_origin, params):
    ''' returns z-plane from (x,y) point and norm parameters (parms)
        params: plane cartesian parameters
        plane_origin: origin point of the plane
    '''
    # # pick the origin point of the new plane at (0,0,c) equivalant to (0,0,1)
    # origin_point_gt = np.array([0.0, 0.0, plane(0,0,plane_final_params)])

    # find the normal vector using the cross product between two unit vectors
    # in the direction of x and y of the new plane
    v1 = np.array([1,0,cartPlane(1,0,params)]) - plane_origin
    v2 = np.array([0,1,cartPlane(0,1,params)]) - plane_origin
    norm_vector = np.array(np.cross(v1, v2))
    # normalise norm vector on the fitted plane
    norm_vector = norm_vector * 1.0 / np.linalg.norm(norm_vector)

    # find the dot product between the norm and origin point sitting on the
    # new plane d = -(ax + by + cz)
    d_gt = -origin_point_gt.dot(v_norm_fit_plane)

    # Point-normal form and general form of the equation of a plane
    # ax + by + cz + d = 0 , where (a,b,c) is the normal vector
    # zz_gt = (-v_norm_fit_plane[0] * xx - v_norm_fit_plane[1] * yy - d_gt) * 1. /v_norm_fit_plane[2]
    # print('zz_gt {}'.format(zz_gt))
    zz_gt = plane(xx, yy, plane_final_params)
    # print('zz_gt {}'.format(zz_gt))

    # plot the new plane surface
    ax.plot_surface(xx, yy, zz_gt, alpha=0.5, color=[0,1,0])

def error(params, points):
    ''' returns the accumaletd absolute error between points (x,y,z) and
    a plane with parameters (params)
    '''
    result = 0
    for (x,y,z) in points:
        plane_z = cartPlane(x, y, params)
        diff = abs(plane_z - z)
        result += diff**2
    return result


def fitPlanePoints(points):

    # optimization function - the error between gt_points and fitted plane
    opt_func = partial(error, points=points)
    # initialize plane equation parameters z = a*x + b*y + c with zeros
    plane_initial_params = [0, 0, 0]
    # final parameter result values
    res = scipy.optimize.minimize(opt_func, plane_initial_params)

    return res.x


def resampleIsotropic(sitk_image, spacing_new=(1.0,1.0,1.0)):

    resampleFilter =sitk.ResampleImageFilter()
    resampleFilter.SetReferenceImage(sitk_image)
    resampleFilter.SetOutputSpacing(spacing_new)
    # (sitkBSpline, sitkNearestNeighbor, sitkBlackmanWindowedSinc,
    # sitkCosineWindowedSinc, sitkLinear, sitkGaussian,
    # sitkHammingWindowedSinc, sitkLanczosWindowedSinc,
    # sitkWelchWindowedSinc, )
    resampleFilter.SetInterpolator(sitk.sitkBSpline)

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


###############################################################################
############################ extract 2D plane  ################################
###############################################################################
def extractMask(sitk_image3d,sitk_image2d):
    ''' extracts a mask image from sitk_image2d with the same size as
    sitk_image3d
    '''
    sitk_mask3d = sitk_image3d * 0
    image2d_size = sitk_image2d.GetSize()
    image3d_size = sitk_image3d.GetSize()

    # print(image2d_size, image3d_size)

    for z in range(image2d_size[2]):
     for y in range(image2d_size[1]):
        for x in range(image2d_size[0]):

            xx,yy,zz = sitk_image2d.TransformIndexToPhysicalPoint((x,y,z))
            xx, yy, zz = sitk_mask3d.TransformPhysicalPointToIndex((xx,yy,zz))

            if (xx<0) or (yy<0) or (zz<0): continue
            if (xx>=image3d_size[0]) or (yy>=image3d_size[1]) or (zz>=image3d_size[2]): continue

            sitk_mask3d.SetPixel(xx,yy,zz,1)

    return sitk_mask3d



def get2DPlaneFrom3DImageGivenTwoNiftis(sitk_image3d,sitk_image2d):
    ''' This function returns the plane equation from a 3D volume nifti giving another 2D plane nifti image -> output: the ground truth plane
    '''
    size_image2d = sitk_image2d.GetSize()
    # extract few points to fit the plane equation
    points_image2d = [(0,0,size_image2d[0],size_image2d[0]),
                      (0,size_image2d[1],0,size_image2d[1]),
                      (0,0,0,0)]

    ## ranodm points
    # number_of_points = 10
    # points_image2d = (np.random.randint(size_image2d[0],size=number_of_points),
    #                   np.random.randint(size_image2d[1],size=number_of_points),
    #                   np.random.randint(size_image2d[2],size=number_of_points))
    # # [print(p) for p in zip(*points_image2d)]
    # [print(np.array(p)) for p in zip(points_image2d.T)]
    points_image2d = [np.array(p).astype('float') for p in zip(*points_image2d)]
    # print('points_image2d\n',points_image2d)
    points_physical = [sitk_image2d.TransformContinuousIndexToPhysicalPoint(index=p) for p in points_image2d]
    # print('points_physical\n',points_physical)
    # [print(p) for p in points_physical]
    points_image3d = [sitk_image3d.TransformPhysicalPointToContinuousIndex(p) for p in points_physical]
    # print('points_image3d\n',points_image3d)
    # print('extract mask image')
    sitk_mask3d = extractMask(sitk_image3d,sitk_image2d)
    sitk.WriteImage(sitk_mask3d, 'sitk_mask3d.nii.gz')
    # print('points_image3d')
    # print(points_image3d)

    return fitPlanePoints(points_image3d)

# def extract2DPlaneFrom3DImageGiven2DImage(sitk_image3d,
#                                           sitk_image2d,
#                                           plane_size):
#     ''' This function returns the plane equation from a 3D volume nifti giving another 2D plane nifti image -> output: the ground truth plane
#     '''
#     size_image2d = sitk_image2d.GetSize()

#     ###########################################################################
#     # extract a xy-grid every 1mm from the sitk_image3d
#     # first extract points on thecenter lines of + on the plane
#     # image_2d coordinates
#     centerPoint = [int(i/2) for i in size_image2d]
#     point_x_positive = (size_image2d[0]-1, centerPoint[1], centerPoint[2])
#     point_y_positive = (centerPoint[0], size_image2d[1]-1, centerPoint[2])

#     # physical coordinates
#     centerPoint = sitk_image2d.TransformContinuousIndexToPhysicalPoint(
#                         centerPoint)
#     point_x_positive = sitk_image2d.TransformContinuousIndexToPhysicalPoint(
#                         point_x_positive)
#     point_y_positive = sitk_image2d.TransformContinuousIndexToPhysicalPoint(
#                         point_y_positive)
#     # unit vectors
#     unit_vector_x_positive = [point_x_positive[0] - centerPoint[0],
#                               point_x_positive[1] - centerPoint[1],
#                               point_x_positive[2] - centerPoint[2]]
#     unit_vector_y_positive = [point_y_positive[0] - centerPoint[0],
#                               point_y_positive[1] - centerPoint[1],
#                               point_y_positive[2] - centerPoint[2]]
#     # normalise unit vectors
#     unit_vector_x_positive = normalizeUnitVector(unit_vector_x_positive)
#     unit_vector_y_positive = normalizeUnitVector(unit_vector_y_positive)

#     unit_vector_z_positive = np.cross(unit_vector_x_positive,
#                                       unit_vector_y_positive)
#     unit_vector_z_positive = normalizeUnitVector(unit_vector_z_positive)


#     return sampleGrid(sitk_image3d, centerPoint,
#                      unit_vector_x_positive,
#                      unit_vector_y_positive,
#                      unit_vector_z_positive,
#                      plane_size,
#                      spacing=(2,2,2))


# def prepareGroundTruthPlane(sitk_image3d,sitk_image2d,
#                             origin_point3d,plane_size):
#     ''' This function extracts the ground truth plane
#         Returns:
#             plane in the norm form
#             corner points of the plane
#     '''
#     size_image2d = sitk_image2d.GetSize()
#     size_image3d = sitk_image3d.GetSize()

#     origin_point3d_physical = np.array(sitk_image3d.TransformContinuousIndexToPhysicalPoint(origin_point3d))

#     # print('origin_point3d ', origin_point3d)
#     # print('origin_point3d_physical ', origin_point3d_physical)

#     # sitk_image2d coordinates
#     center_point2d = [int(i/2) for i in size_image2d]
#     pointx2d = (center_point2d[0]+plane_size[0]/2, center_point2d[1], center_point2d[2])
#     pointy2d = (center_point2d[0], center_point2d[1]+plane_size[1]/2, center_point2d[2])
#     # physical coordinates
#     center_point2d_physical = sitk_image2d.TransformContinuousIndexToPhysicalPoint(
#                         center_point2d)
#     pointx2d_physical = sitk_image2d.TransformContinuousIndexToPhysicalPoint(
#                         pointx2d)
#     pointy2d_physical = sitk_image2d.TransformContinuousIndexToPhysicalPoint(
#                         pointy2d)
#     # find the correspondant plane in physical space
#     # new plane d = -(ax + by + cz)
#     v1 = np.array(pointx2d_physical) - np.array(center_point2d_physical)
#     v2 = np.array(pointy2d_physical) - np.array(center_point2d_physical)
#     norm_vector = np.array(np.cross(v1, v2))
#     # normalise norm vector
#     norm_vector = norm_vector * 1.0 / np.linalg.norm(norm_vector)

#     # sample a 2d grid of size[x,y] in xy-directions of the image_3d
#     pointx3d = (size_image3d[0]-plane_size[0]/2, origin_point3d[1], origin_point3d[2])
#     pointy3d = (origin_point3d[0], size_image3d[1]-plane_size[1]/2, origin_point3d[2])
#     # physical coordinates
#     pointx3d_physical = sitk_image3d.TransformContinuousIndexToPhysicalPoint(
#                         pointx3d)
#     # pointy3d_physical = sitk_image3d.TransformContinuousIndexToPhysicalPoint(
#     #                     pointy3d)
#     # project points on plane
#     origin_point3d_projected = projectPointOnPlane(origin_point3d_physical, norm_vector, center_point2d_physical)
#     pointx3d_projected = projectPointOnPlane(pointx3d_physical, norm_vector, center_point2d_physical)
#     # pointy3d_projected = projectPointOnPlane(pointy3d_physical, norm_vector, center_point2d_physical)

#     # unit vectors
#     unit_vector_x_positive =[
#         pointx3d_projected[0] - origin_point3d_projected[0],
#         pointx3d_projected[1] - origin_point3d_projected[1],
#         pointx3d_projected[2] - origin_point3d_projected[2]]

#     # unit_vector_y_positive = [
#     #     pointy3d_projected[0] - origin_point3d_projected[0],
#     #     pointy3d_projected[1] - origin_point3d_projected[1],
#     #     pointy3d_projected[2] - origin_point3d_projected[2]]

#     # normalize unit vectors
#     unit_vector_x_positive = np.array(unit_vector_x_positive) / np.linalg.norm(unit_vector_x_positive)
#     # unit_vector_y_positive = np.array(unit_vector_y_positive) / np.linalg.norm(unit_vector_y_positive)

#     unit_vector_y_positive = np.cross(norm_vector, unit_vector_x_positive)
#     unit_vector_y_positive = np.array(unit_vector_y_positive) / np.linalg.norm(unit_vector_y_positive)

#     unit_vector_z_positive = norm_vector

#     grid_2d, corner_points = sampleGrid(sitk_image3d,origin_point3d_projected,
#                                        unit_vector_x_positive,
#                                        unit_vector_y_positive,
#                                        unit_vector_z_positive,
#                                        plane_size,
#                                        spacing=(2,2,2))

#     return grid_2d, corner_points, norm_vector, origin_point3d_projected

def getGroundTruthPlane(sitk_image3d, sitk_image2d, origin, plane_size,
                        spacing=(1,1,1)):
    ''' This function extracts the ground truth plane
        Returns:
            plane in the norm form
            corner points of the plane
    '''
    size_image2d = sitk_image2d.GetSize()
    size_image3d = sitk_image3d.GetSize()
    # -------------------------------------------------------------------------
    ## first step is to find the equation of the ground truth plane with
    ## respect to the 3d image coordinates
    # extract three points in sitk_image2d coordinates
    center2d = [int(i/2) for i in size_image2d]
    pointx2d = (center2d[0]+plane_size[0]/2, center2d[1], center2d[2])
    pointy2d = (center2d[0], center2d[1]+plane_size[1]/2, center2d[2])
    # transform these points to physical coordinates
    center2d_physical = sitk_image2d.TransformContinuousIndexToPhysicalPoint(                center2d)
    pointx2d_physical = sitk_image2d.TransformContinuousIndexToPhysicalPoint(
                        pointx2d)
    pointy2d_physical = sitk_image2d.TransformContinuousIndexToPhysicalPoint(
                        pointy2d)
    # transform these points to physical coordinates
    center3d = sitk_image3d.TransformPhysicalPointToContinuousIndex(center2d_physical)
    pointx3d = sitk_image3d.TransformPhysicalPointToContinuousIndex(pointx2d_physical)
    pointy3d = sitk_image3d.TransformPhysicalPointToContinuousIndex(pointy2d_physical)

    # find the correspondant plane in physical space
    # new plane d = -(ax + by + cz)
    # here the plane is defined using a plane norm and origin point
    # normalise norm vector
    v1 = np.array(pointx3d) - np.array(center3d)
    v2 = np.array(pointy3d) - np.array(center3d)
    plane_norm = np.cross(v1, v2)
    # normalize the norm vector
    plane_norm = normalizeUnitVector(plane_norm)
    # -------------------------------------------------------------------------
    # plane origin is defined by projecting the 3d origin on the plane
    plane_origin, d = projectPointOnPlane(origin, plane_norm, center3d)
    # plane parameters
    plane_params = [np.rad2deg(np.arccos(plane_norm[0])),
                    np.rad2deg(np.arccos(plane_norm[1])),
                    np.rad2deg(np.arccos(plane_norm[2])),
                    d]
    # -------------------------------------------------------------------------
    plane_origin_physical = np.array(sitk_image3d.TransformContinuousIndexToPhysicalPoint(plane_origin))
    # find point in x-direction of the 3d volume to sample in this direction
    pointx = (origin[0] + plane_size[0]/2, origin[1], origin[2])
    pointx_proj, _ = projectPointOnPlane(pointx, plane_norm, plane_origin)
    pointx_proj_physical = np.array(sitk_image3d.TransformContinuousIndexToPhysicalPoint(pointx_proj))
    vectorx = normalizeUnitVector(pointx_proj_physical - plane_origin_physical)
    # z-direction
    # find point in the new positive z-direction (plane norm)
    pointz = plane_origin + (plane_size[2]/2) * plane_norm
    pointz_physical = np.array(sitk_image3d.TransformContinuousIndexToPhysicalPoint(pointz))
    vectorz = normalizeUnitVector(pointz_physical - plane_origin_physical)
    # y-direction
    vectory = np.cross(vectorz, vectorx)
    vectory = normalizeUnitVector(vectory)
    # sample a grid int he calculated directions
    grid, points = sampleGrid(sitk_image3d,
                              plane_origin,
                              vectorx,
                              vectory,
                              vectorz,
                              plane_size,
                              spacing=spacing)

    grid = ndimage.median_filter(grid, size=3)
    grid = ndimage.spline_filter(grid)#

    # set_trace()
    # import matplotlib.pyplot as plt
    # plt.imshow(grid[:,:,5], cmap=plt.cm.Greys_r)
    # plt.show()
    # set_trace()

    return grid, plane_norm, plane_origin, plane_params, points

# def getGroundTruthPlane(sitk_image3d,sitk_image2d,
#                         origin_point3d,plane_size,spacing=(1,1,1)):
#     ''' This function extracts the ground truth plane
#         Returns:
#             plane in the norm form
#             corner points of the plane
#     '''
#     size_image2d = sitk_image2d.GetSize()
#     size_image3d = sitk_image3d.GetSize()

#     origin_point3d_physical = np.array(sitk_image3d.TransformContinuousIndexToPhysicalPoint(origin_point3d))

#     # extract three points in sitk_image2d coordinates
#     center_point2d = [int(i/2) for i in size_image2d]

#     pointx2d = (center_point2d[0]+plane_size[0]/2,
#                 center_point2d[1],
#                 center_point2d[2])

#     pointy2d = (center_point2d[0],
#                 center_point2d[1]+plane_size[1]/2,
#                 center_point2d[2])
#     # transform these points to physical coordinates
#     center_point2d_physical = sitk_image2d.TransformContinuousIndexToPhysicalPoint(center_point2d)
#     pointx2d_physical = sitk_image2d.TransformContinuousIndexToPhysicalPoint(
#                         pointx2d)
#     pointy2d_physical = sitk_image2d.TransformContinuousIndexToPhysicalPoint(
#                         pointy2d)
#     # find the correspondant plane in physical space
#     # new plane d = -(ax + by + cz)
#     v1 = np.array(pointx2d_physical) - np.array(center_point2d_physical)
#     v2 = np.array(pointy2d_physical) - np.array(center_point2d_physical)
#     plane_norm = np.array(np.cross(v1, v2))
#     # -------------------------------------------------------------------------
#     # here the plane is defined using a plane norm and origin point
#     # normalise norm vector
#     plane_norm = normalizeUnitVector(plane_norm)
#     # plane origin is defined by projecting the 3d origin on the plane
#     plane_origin = projectPointOnPlane(origin_point3d_physical,
#                                        plane_norm,
#                                        center_point2d_physical)
#     # -------------------------------------------------------------------------
#     # find point in x-direction of the 3d volume to sample in this direction
#     # sample a 2d grid of size[x,y] in xy-directions of the image_3d
#     pointx3d = (size_image3d[0]-plane_size[0]/2,
#                 origin_point3d[1],
#                 origin_point3d[2])
#     # physical coordinates
#     pointx3d_physical = sitk_image3d.TransformContinuousIndexToPhysicalPoint(
#                         pointx3d)
#     pointx3d_projected = projectPointOnPlane(pointx3d_physical,
#                                              plane_norm,
#                                              center_point2d_physical)
#     # new plane coordinate unit vectors
#     unit_vector_x_positive =[pointx3d_projected[0]-plane_origin[0],
#                             pointx3d_projected[1]-plane_origin[1],
#                             pointx3d_projected[2]-plane_origin[2]]
#     unit_vector_y_positive = np.cross(plane_norm, unit_vector_x_positive)
#     # normalize unit vectors
#     unit_vector_x_positive = normalizeUnitVector(unit_vector_x_positive)
#     unit_vector_y_positive = normalizeUnitVector(unit_vector_y_positive)
#     unit_vector_z_positive = np.copy(plane_norm)

#     grid_2d, corner_points = sampleGrid(sitk_image3d,plane_origin,
#                                        unit_vector_x_positive,
#                                        unit_vector_y_positive,
#                                        unit_vector_z_positive,
#                                        plane_size,
#                                        spacing=spacing)
#     # plane equation ax+by+cz=d , where norm = (a,b,c), and d=a*x0+b*y0+c*z0
#     a, b, c = plane_norm[0], plane_norm[1], plane_norm[2]
#     x0, y0, z0 = plane_origin[0], plane_origin[1], plane_origin[2]

#     d = a*x0 + b*y0 + c*z0

#     plane_params = [np.rad2deg(np.arccos(a)),
#                     np.rad2deg(np.arccos(b)),
#                     np.rad2deg(np.arccos(c)),
#                     d]

#     return grid_2d, plane_norm, plane_origin, plane_params, corner_points

def getPlane(sitk_image3d, origin, plane_params, plane_size, spacing=(1,1,1)):
    ''' Get a plane from a 3d nifti image using its norm form
    '''
    # plane equation ax+by+cz=d , where norm = (a,b,c), and d=a*x0+b*y0+c*z0
    a, b, c, d = [np.cos(np.deg2rad(plane_params[0])),
                  np.cos(np.deg2rad(plane_params[1])),
                  np.cos(np.deg2rad(plane_params[2])),
                  plane_params[3]]
    # find plane norm vector
    plane_norm = np.array((a,b,c))
    # get transformation and origin
    origin3d = np.array(sitk_image3d.GetOrigin())
    direction = np.array(sitk_image3d.GetDirection())
    transformation = np.array(direction.reshape(3,3))
    transformation_inv = np.linalg.inv(transformation)

    # find plane origin
    plane_origin = origin + d * plane_norm
    plane_origin_physical = plane_origin.dot(transformation_inv) + origin3d
    # plane_origin_physical = np.array(sitk_image3d.TransformContinuousIndexToPhysicalPoint(plane_origin))
    # find point in x-direction of the 3d volume to sample in this direction
    pointx = (origin[0] + plane_size[0]/2, origin[1], origin[2])
    pointx_proj, _ = projectPointOnPlane(pointx, plane_norm, plane_origin)
    pointx_proj_physical = pointx_proj.dot(transformation_inv) + origin3d
    # pointx_proj_physical = np.array(sitk_image3d.TransformContinuousIndexToPhysicalPoint(pointx_proj))
    vectorx = normalizeUnitVector(pointx_proj_physical - plane_origin_physical)
    # z-direction
    # find point in the new positive z-direction (plane norm)
    pointz = plane_origin + (plane_size[2]/2) * plane_norm
    pointz_physical = pointz.dot(transformation_inv) + origin3d
    # pointz_physical = np.array(sitk_image3d.TransformContinuousIndexToPhysicalPoint(pointz))
    vectorz = normalizeUnitVector(pointz_physical - plane_origin_physical)
    # y-direction
    vectory = np.cross(vectorz, vectorx)
    vectory = normalizeUnitVector(vectory)
    # sample a grid in the calculated directions
    grid, points = sampleGrid(sitk_image3d,
                              plane_origin,
                              vectorx,
                              vectory,
                              vectorz,
                              plane_size,
                              spacing=spacing)

    # filter grid
    # blurred_grid = ndimage.gaussian_filter(grid, sigma=3)
    # very_blurred = ndimage.gaussian_filter(face, sigma=5)
    grid = ndimage.uniform_filter(grid, size=3)
    grid = ndimage.spline_filter(grid)#

    # logger.info('plane_norm {}'.format(np.around(plane_norm,2)))
    # logger.info('plane_origin {}'.format(np.around(plane_origin,2)))

    return grid, plane_norm, plane_origin, plane_params, points



# def getPlane(sitk_image3d, origin_image3d,
#         plane_params, plane_size, spacing=(1,1,1)):
#     ''' Get a plane from a 3d nifti image using its norm form
#     '''
#     # plane equation ax+by+cz=d , where norm = (a,b,c), and d=a*x0+b*y0+c*z0
#     a, b, c, d = [np.cos(np.deg2rad(plane_params[0])),
#                   np.cos(np.deg2rad(plane_params[1])),
#                   np.cos(np.deg2rad(plane_params[2])),
#                   plane_params[3]]
#     print('a {}, b {}, c {}, d {}'.format(a, b, c, d))

#     plane_norm = np.array((a,b,c))

#     # get point on the new plane
#     origin_image3d_physical = sitk_image3d.TransformContinuousIndexToPhysicalPoint(origin_image3d)
#     x, y = origin_image3d_physical[0:2]
#     z = (d - a*x - b*y) / c
#     point_plane_new = (x,y,z)
#     # project the 3d origin on the new plane to get the new plane origin
#     plane_origin_new = projectPointOnPlane(origin_image3d_physical, plane_norm, point_plane_new)
#     # select a random point on the x-axis of image3d
#     pointx_image3d = (origin_image3d[0] + plane_size[0]/2,
#                       origin_image3d[1],
#                       origin_image3d[2])
#     pointx3d_physical = sitk_image3d.TransformContinuousIndexToPhysicalPoint(pointx_image3d)
#     pointx_projected = projectPointOnPlane(pointx3d_physical,
#                                            plane_norm, point_plane_new)
#     # unit vectors
#     unit_vector_x_positive = np.array((pointx_projected[0]-plane_origin_new[0],
#                                     pointx_projected[1] - plane_origin_new[1],
#                                     pointx_projected[2] - plane_origin_new[2]))
#     # find the y-axis
#     unit_vector_y_positive = np.cross(plane_norm, unit_vector_x_positive)
#     # copy the z-axis from plane norm
#     unit_vector_z_positive = np.array(plane_norm)
#     # normalize unit vectors
#     unit_vector_x_positive = normalizeUnitVector(unit_vector_x_positive)
#     unit_vector_y_positive = normalizeUnitVector(unit_vector_y_positive)
#     unit_vector_z_positive = normalizeUnitVector(unit_vector_z_positive)
#     # get the grid and corner points
#     grid_2d, corner_points = sampleGrid(sitk_image3d, plane_origin_new,
#                                        unit_vector_x_positive,
#                                        unit_vector_y_positive,
#                                        unit_vector_z_positive,
#                                        plane_size,
#                                        spacing=spacing)
#     # # get new plane origin and params
#     # x0, y0, z0 = plane_origin_new[0], plane_origin_new[1], plane_origin_new[2]
#     # d = a*x0 + b*y0 + c*z0

#     # plane_params = [np.rad2deg(np.arccos(a)),
#     #                 np.rad2deg(np.arccos(b)),
#     #                 np.rad2deg(np.arccos(c)),
#     #                 d]
#     # print('a {}, b {}, c {}, d {}'.format(a, b, c, d))
#     return grid_2d, plane_norm, plane_origin_new, plane_params, corner_points
# def getInitialPlane(sitk_image3d, plane_size,origin_point=None):
#     ''' This function extracts initial plane, which is the mid xy-plane
#         Returns:
#             plane in the norm form
#     '''

#     # extract three points in sitk_image3d coordinates
#     if not origin_point:
#         origin_point = [int(i/2) for i in sitk_image3d.GetSize()]

#     pointx = (origin_point[0] + 1, origin_point[1], origin_point[2])
#     pointy = (origin_point[0], origin_point[1] + 1, origin_point[2])
#     pointz = (origin_point[0], origin_point[1], origin_point[2]+1)
#     # physical coordinates
#     plane_origin = sitk_image3d.TransformContinuousIndexToPhysicalPoint(origin_point)
#     pointx_physical = sitk_image3d.TransformContinuousIndexToPhysicalPoint(
#                         pointx)
#     pointy_physical = sitk_image3d.TransformContinuousIndexToPhysicalPoint(
#                         pointy)
#     pointz_physical = sitk_image3d.TransformContinuousIndexToPhysicalPoint(
#                         pointz)
#     # find the correspondant plane in physical space
#     # new plane d = -(ax + by + cz)
#     unit_vector_x_positive = np.array(pointx_physical) - np.array(plane_origin)
#     unit_vector_y_positive = np.array(pointy_physical) - np.array(plane_origin)
#     unit_vector_z_positive = np.array(pointz_physical) - np.array(plane_origin)
#     # normalise vectors
#     # plane_norm = norm_vector * 1.0 / np.linalg.norm(norm_vector)
#     unit_vector_x_positive = unit_vector_x_positive * 1.0 / np.linalg.norm(unit_vector_x_positive)
#     unit_vector_y_positive = unit_vector_y_positive * 1.0 / np.linalg.norm(unit_vector_y_positive)
#     unit_vector_z_positive = unit_vector_z_positive * 1.0 / np.linalg.norm(unit_vector_z_positive)
#     # sample a grid of size[plane_size]
#     plane_norm = unit_vector_z_positive
#     grid_2d, corner_points = sampleGrid(sitk_image3d,plane_origin,
#                                        unit_vector_x_positive,
#                                        unit_vector_y_positive,
#                                        unit_vector_z_positive,
#                                        plane_size,
#                                        spacing=(2,2,2))


#     return grid_2d, plane_norm, plane_origin, corner_points

def getInitialPlane(sitk_image3d, plane_size,
                    origin=None, spacing=(1,1,1)):
    ''' This function extracts initial plane, which is the mid xy-plane
        Returns:
            plane in the norm form
    '''

    # origin point of the plane
    size_image3d = sitk_image3d.GetSize()
    if origin:
        origin = [int(i/2) for i in size_image3d]

    # get initial plane in the standard xy-directions
    d = 0
    plane_norm = np.array([0,0,1])
    plane_origin = deepcopy(origin)
    plane_params = [np.rad2deg(np.arccos(plane_norm[0])),
                    np.rad2deg(np.arccos(plane_norm[1])),
                    np.rad2deg(np.arccos(plane_norm[2])),
                    d]

    # find directions in the physical space to sample a grid
    pointx = (origin[0] + plane_size[0]/2, origin[1], origin[2])
    pointy = (origin[0], origin[1] + plane_size[1]/2, origin[2])
    pointz = (origin[0], origin[1], origin[2]+plane_size[2]/2)
    # physical coordinates
    origin_physical = np.array(sitk_image3d.TransformContinuousIndexToPhysicalPoint(origin))
    pointx_physical = np.array(sitk_image3d.TransformContinuousIndexToPhysicalPoint(pointx))
    pointy_physical = np.array(sitk_image3d.TransformContinuousIndexToPhysicalPoint(pointy))
    pointz_physical = np.array(sitk_image3d.TransformContinuousIndexToPhysicalPoint(pointz))
    # direction vectors
    vectorx = normalizeUnitVector(pointx_physical - origin_physical)
    vectory = normalizeUnitVector(pointy_physical - origin_physical)
    vectorz = normalizeUnitVector(pointz_physical - origin_physical)
    # sample a grid of size[plane_size]
    grid, points = sampleGrid(sitk_image3d,
                              plane_origin,
                              vectorx,
                              vectory,
                              vectorz,
                              plane_size,
                              spacing=spacing)

    # set_trace()
    # import matplotlib.pyplot as plt
    # plt.imshow(grid[:,:,5], cmap=plt.cm.Greys_r)
    # plt.show()
    # set_trace()

    return grid, plane_norm, plane_origin, plane_params, points
# def getInitialPlane(sitk_image3d, plane_size,
#                     origin_point=None, spacing=(1,1,1)):
#     ''' This function extracts initial plane, which is the mid xy-plane
#         Returns:
#             plane in the norm form
#     '''

#     # extract three points in sitk_image3d coordinates
#     if not origin_point:
#         origin_point = [int(i/2) for i in sitk_image3d.GetSize()]

#     pointx = (origin_point[0] + 1, origin_point[1], origin_point[2])
#     pointy = (origin_point[0], origin_point[1] + 1, origin_point[2])
#     pointz = (origin_point[0], origin_point[1], origin_point[2]+1)
#     # physical coordinates
#     plane_origin = sitk_image3d.TransformContinuousIndexToPhysicalPoint(origin_point)
#     pointx_physical = sitk_image3d.TransformContinuousIndexToPhysicalPoint(
#                         pointx)
#     pointy_physical = sitk_image3d.TransformContinuousIndexToPhysicalPoint(
#                         pointy)
#     pointz_physical = sitk_image3d.TransformContinuousIndexToPhysicalPoint(
#                         pointz)
#     # find the correspondant plane in physical space
#     # new plane d = -(ax + by + cz)
#     unit_vector_x_positive = np.array(pointx_physical) - np.array(plane_origin)
#     unit_vector_y_positive = np.array(pointy_physical) - np.array(plane_origin)
#     unit_vector_z_positive = np.array(pointz_physical) - np.array(plane_origin)
#     # normalise vectors
#     unit_vector_x_positive = normalizeUnitVector(unit_vector_x_positive)
#     unit_vector_y_positive = normalizeUnitVector(unit_vector_y_positive)
#     unit_vector_z_positive = normalizeUnitVector(unit_vector_z_positive)

#     # sample a grid of size[plane_size]
#     grid_2d, corner_points = sampleGrid(sitk_image3d,plane_origin,
#                                        unit_vector_x_positive,
#                                        unit_vector_y_positive,
#                                        unit_vector_z_positive,
#                                        plane_size,
#                                        spacing=spacing)

#     # plane equation ax+by+cz=d , where norm = (a,b,c), and d=a*x0+b*y0+c*z0
#     plane_norm = unit_vector_z_positive
#     a, b, c = plane_norm[0], plane_norm[1], plane_norm[2]
#     x0, y0, z0 = plane_origin[0], plane_origin[1], plane_origin[2]

#     d = a*x0 + b*y0 + c*z0

#     plane_params = [np.rad2deg(np.arccos(a)),
#                     np.rad2deg(np.arccos(b)),
#                     np.rad2deg(np.arccos(c)),
#                     d]

#     return grid_2d, plane_norm, plane_origin, plane_params, corner_points

def normalizeUnitVector(vector):
    ''' return the normalized unit vector '''
    return np.array(vector) / np.linalg.norm(vector)

def projectPointOnPlane(point, norm, point_in_plane):
    # project the origin point in 3d on the plane
    # The distance from origin_3d to the plane, is simply the length of the
    # projection of a vector between this point and any poing ton the plane
    # onto the unit normal vector.
    # Since norm is length one, this distance is simply the absolute value of
    # the dot product d = v⋅n. Finally add d to the 3d origin point
    v = np.array(point_in_plane) - point
    # distance to plane
    d_plane = norm.dot(v)
    # projected point
    proj_point = point + d_plane*norm
    return proj_point, d_plane

# def projectPointOnPlane(point, norm, point_in_plane):
#     # project the origin point in 3d on the plane
#     # The distance from origin_3d to the plane, is simply the length of the
#     # projection of a vector between this point and any poing ton the plane
#     # onto the unit normal vector.
#     # Since norm is length one, this distance is simply the absolute value of
#     # the dot product d = v⋅n. Finally add d to the 3d origin point
#     v = np.array(point_in_plane) - point
#     d_plane = norm.dot(v)
#     return point + d_plane*norm


import timeit

## first implmentation with 3 nested loops
# def sampleGrid(sitk_image3d, origin_point,
#               directionx, directiony, directionz,
#               plane_size, spacing):

#     # get plane size - take care of odd and even sizes
#     # x-axis range
#     if np.mod(plane_size[0],2):
#         size_xp = (int)(plane_size[0]/2) + 1
#         size_xn = (int)(plane_size[0]/2)
#     else:
#         size_xp = size_xn = (int)(plane_size[0]/2)
#     # y-axis range
#     if np.mod(plane_size[1],2):
#         size_yp = (int)(plane_size[1]/2) + 1
#         size_yn = (int)(plane_size[1]/2)
#     else:
#         size_yp = size_yn = (int)(plane_size[1]/2)
#     # z-axis range
#     if np.mod(plane_size[2],2):
#         size_zp = (int)(plane_size[2]/2) + 1
#         size_zn = (int)(plane_size[2]/2)
#     else:
#         size_zp = size_zn = (int)(plane_size[2]/2)

#     # initialize the grid
#     grid = np.zeros(plane_size).astype('float')
#     size_xn, size_xp = spacing[0] * size_xn, spacing[0] * size_xp
#     size_yn, size_yp = spacing[1] * size_yn, spacing[1] * size_yp
#     size_zn, size_zp = spacing[2] * size_zn, spacing[2] * size_zp

#     corner_points = []

#     start = time.time()

#     image3d = sitk.GetArrayFromImage(sitk_image3d)
#     size3d = sitk_image3d.GetSize()

#     for i, x in enumerate(range(-size_xn,size_xp,spacing[0])):
#         for j, y in enumerate(range(-size_yn,size_yp,spacing[1])):
#             for k, z in enumerate(range(-size_zn,size_zp,spacing[2])):
#                 point_physical = origin_point + x * directionx + y * directiony + z * directionz
#                 # extract the exact point on the plane
#                 # print('xyz ', point_physical)
#                 point = sitk_image3d.TransformPhysicalPointToIndex(point_physical)

#                 # try:
#                 #     pixel_value = sitk_image3d.GetPixel(point)
#                 #     # check if point is ouside image
#                 #     # set_trace()
#                 if (not any([(i<0 or i>=size3d[j])
#                             for j,i in enumerate(point)])):
#                     try:
#                         grid[i,j,k] = image3d[point[2],point[1],point[0]]
#                         # grid[i,j,k] = pixel_value
#                     except:
#                         print('outside grid')
#                 # except:
#                 else:
#                     pass

#                 # store corner points
#                 if (((x == -size_xn) and (y == -size_yn) and (z==0))
#                     or
#                     ((x == -size_xn) and (y == size_yp-spacing[1]) and (z==0))
#                     or
#                     ((x == size_xp-spacing[0]) and (y == -size_yn) and (z==0))
#                     or
#                     ((x == size_xp-spacing[0]) and (y == size_yp-spacing[1]) and (z==0))):
#                     corner_points.append(list(point_physical))

#     end = time.time()
#     print('time elapsed', end - start)

#     return grid, sorted(corner_points)

def sampleGrid(sitk_image3d, origin_point,
              directionx, directiony, directionz,
              plane_size, spacing):

    # get plane size - take care of odd and even sizes
    # x-axis range
    if np.mod(plane_size[0],2):
        size_xp = (int)(plane_size[0]/2) + 1
        size_xn = (int)(plane_size[0]/2)
    else:
        size_xp = size_xn = (int)(plane_size[0]/2)
    # y-axis range
    if np.mod(plane_size[1],2):
        size_yp = (int)(plane_size[1]/2) + 1
        size_yn = (int)(plane_size[1]/2)
    else:
        size_yp = size_yn = (int)(plane_size[1]/2)
    # z-axis range
    if np.mod(plane_size[2],2):
        size_zp = (int)(plane_size[2]/2) + 1
        size_zn = (int)(plane_size[2]/2)
    else:
        size_zp = size_zn = (int)(plane_size[2]/2)

    # scale plane size with spacing
    size_xn, size_xp = spacing[0] * size_xn, spacing[0] * size_xp
    size_yn, size_yp = spacing[1] * size_yn, spacing[1] * size_yp
    size_zn, size_zp = spacing[2] * size_zn, spacing[2] * size_zp

    # find range vectors of plane size
    x_physical = np.arange(-size_xn,size_xp,spacing[0])
    y_physical = np.arange(-size_yn,size_yp,spacing[1])
    z_physical = np.arange(-size_zn,size_zp,spacing[2])
    # find the correspondant grids
    x_grid, y_grid, z_grid = np.meshgrid(x_physical, y_physical, z_physical)
    # multiply with directions
    x_grid = np.multiply.outer(x_grid, directionx)
    y_grid = np.multiply.outer(y_grid, directiony)
    z_grid = np.multiply.outer(z_grid, directionz)
    # find a grid of the origin of sampling plane (center point)
    origin_point_physical = sitk_image3d.TransformContinuousIndexToPhysicalPoint(origin_point)
    origin_grid = np.multiply.outer(np.ones(plane_size),origin_point_physical)
    # find final grid values
    grid_final = x_grid + y_grid + z_grid + origin_grid

    # reshape to one vector of 3d points indexes
    points_physical = np.reshape(grid_final,(-1,3))
    # transform points from physical space to sitk_image3d coordinates
    translation = np.array(sitk_image3d.GetOrigin())
    points_physical_translated = points_physical - translation

    direction = np.array(sitk_image3d.GetDirection())
    transformation = np.array(direction.reshape(3,3))
    points_image3d = points_physical_translated.dot(transformation).astype('int')

    # assign indexes outisde the boundary zeros (no negative indexes)
    size3d = sitk_image3d.GetSize()
    x_indexes_outside_image3d = np.logical_or(points_image3d[:,0] < 0.0,
                                              points_image3d[:,0] >= size3d[0])
    y_indexes_outside_image3d = np.logical_or(points_image3d[:,1] < 0.0,
                                              points_image3d[:,1] >= size3d[1])
    z_indexes_outside_image3d = np.logical_or(points_image3d[:,2] < 0.0,
                                              points_image3d[:,2] >= size3d[2])
    all_indexes_outside_image3d = np.logical_or(x_indexes_outside_image3d,
                                                y_indexes_outside_image3d)
    all_indexes_outside_image3d = np.logical_or(all_indexes_outside_image3d,
                                                z_indexes_outside_image3d)
    # replace outlier indexes with zeros
    points_image3d_inboundary = np.copy(points_image3d)
    points_image3d_inboundary[all_indexes_outside_image3d] = [0,0,0]
    # get intensity values from the 3d image
    image3d = sitk.GetArrayFromImage(sitk_image3d).T
    values = image3d[points_image3d_inboundary[:,0],points_image3d_inboundary[:,1],points_image3d_inboundary[:,2]]
    # re-assign zero values for outside indexes
    values[all_indexes_outside_image3d] = 0
    # reshape to plane size
    grid = np.reshape(values, plane_size)
    # select few points
    points = [grid_final[0,0,0,:].tolist(),
              grid_final[-1,0,0,:].tolist(),
              grid_final[0,-1,0,:].tolist(),
              grid_final[0,0,-1,:].tolist(),
              grid_final[-1,-1,-1,:].tolist()]

    return grid, points #sorted(corner_points)



###############################################################################

def sampleGridParallel(sitk_image3d, origin_point,
              directionx, directiony, directionz,
              plane_size, spacing):

    # get plane size - take care of odd and even sizes
    if np.mod(plane_size[0],2):
        size_xp = (int)(plane_size[0]/2) + 1
        size_xn = (int)(plane_size[0]/2)
    else:
        size_xp = size_xn = (int)(plane_size[0]/2)

    if np.mod(plane_size[1],2):
        size_yp = (int)(plane_size[1]/2) + 1
        size_yn = (int)(plane_size[1]/2)
    else:
        size_yp = size_yn = (int)(plane_size[1]/2)

    if np.mod(plane_size[2],2):
        size_zp = (int)(plane_size[2]/2) + 1
        size_zn = (int)(plane_size[2]/2)
    else:
        size_zp = size_zn = (int)(plane_size[0]/2)

    # sample a grid of size[plane_size]
    # rangex = (origin_point[0]-size_xn, origin_point[0]+size_xp)
    # rangey = (origin_point[1]-size_yn, origin_point[1]+size_yp)
    # rangez = (origin_point[2]-size_zn, origin_point[2]+size_zp)

    grid = np.zeros(plane_size[::-1]).astype('float')
    size_xn, size_xp = spacing[0] * size_xn, spacing[0] *size_xp
    size_yn, size_yp = spacing[1] * size_yn, spacing[1] *size_yp
    size_zn, size_zp = spacing[2] * size_zn, spacing[2] *size_zp

    corner_points = []

    for i, x in enumerate(range(-size_xn,size_xp,spacing[0])):
        for j, y in enumerate(range(-size_yn,size_yp,spacing[1])):
            for k, z in enumerate(range(-size_zn,size_zp,spacing[2])):
                point_physical = origin_point + x * directionx + y * directiony + z * directionz
                # extract the exact point on the plane
                # print('xyz ', point_physical)
                point = sitk_image3d.TransformPhysicalPointToIndex(point_physical)

                try:
                    pixel_value = sitk_image3d.GetPixel(point)
                    try:
                        grid[k,j,i] = pixel_value
                    except:
                        print('outside grid')
                    # print('xyz ', point_physical)
                    # print('pixel_value {}, index = {} '.format(pixel_value,point))
                except:
                    pass
                    # print('out ', point)

                # store corner points
                if (((x == -size_xn) and (y == -size_yn) and (z==0)) or
                    ((x == -size_xn) and (y == size_yp-1) and (z==0)) or
                    ((x == size_xp-1) and (y == -size_yn) and (z==0)) or
                    ((x == size_xp-1) and (y == size_yp-1) and (z==0))):
                    corner_points.append(list(point_physical))

    # print('corner_points')
    # print('\n'.join([' '.join(['{}'.format((int)(item)) for item in row])
    #                 for row in corner_points]))

    # print('corner_points_sorted')
    # print('\n'.join([' '.join(['{}'.format((int)(item)) for item in row])
    #                 for row in sorted(corner_points)]))

    return grid, sorted(corner_points)



from IPython.core.debugger import set_trace

def play(sitk_image3d, origin_image3d,
        plane_params, plane_size,
        action, spacing=(1,1,1), change_direction=False):
    ''' play an action and retrieve a new plane
    '''
    # set_trace()
    # plane equation ax+by+cz=d , where norm = (a,b,c), and d=a*x0+b*y0+c*z0
    d = plane_params[-1]
    # result from action here and update plane params
    if change_direction:
        d -= 1
    else:
        d += 1

    plane_params[-1] = d

    return getPlane(sitk_image3d, origin_image3d,
                    plane_params, plane_size, spacing=spacing)

    # # get point on the new plane
    # origin_image3d_physical = sitk_image3d.TransformContinuousIndexToPhysicalPoint(origin_image3d)
    # x, y = origin_image3d_physical[0:2]
    # z = (d - a*x - b*y) / c
    # point_plane_new = (x,y,z)
    # # project the 3d origin on the new plane to get the new plane origin
    # plane_origin_new = projectPointOnPlane(origin_image3d_physical, plane_norm, point_plane_new)
    # # select a random point on the x-axis of image3d
    # pointx_image3d = (origin_image3d[0] + plane_size[0]/2, origin_image3d[1], origin_image3d[2])
    # pointx3d_physical = sitk_image3d.TransformContinuousIndexToPhysicalPoint(pointx_image3d)
    # pointx_projected = projectPointOnPlane(pointx3d_physical, plane_norm, point_plane_new)
    # # unit vectors
    # unit_vector_x_positive = [pointx_projected[0] - plane_origin_new[0],
    #                           pointx_projected[1] - plane_origin_new[1],
    #                           pointx_projected[2] - plane_origin_new[2]]
    # # normalize unit vectors
    # unit_vector_x_positive = np.array(unit_vector_x_positive) / np.linalg.norm(unit_vector_x_positive)
    # # find the y-axis
    # unit_vector_y_positive = np.cross(plane_norm, unit_vector_x_positive)
    # unit_vector_y_positive = np.array(unit_vector_y_positive) / np.linalg.norm(unit_vector_y_positive)

    # # get plane size - take care of odd and even sizes
    # if np.mod(plane_size[0],2):
    #     size_xp = (int)(plane_size[0]/2) + 1
    #     size_xn = (int)(plane_size[0]/2)
    # else:
    #     size_xp = size_xn = (int)(plane_size[0]/2)

    # if np.mod(plane_size[1],2):
    #     size_yp = (int)(plane_size[1]/2) + 1
    #     size_yn = (int)(plane_size[1]/2)
    # else:
    #     size_yp = size_yn = (int)(plane_size[1]/2)

    # if np.mod(plane_size[2],2):
    #     size_zp = (int)(plane_size[2]/2) + 1
    #     size_zn = (int)(plane_size[2]/2)
    # else:
    #     size_zp = size_zn = (int)(plane_size[0]/2)
    # # # sample a grid of size[plane_size]
    # # rangex = (plane_origin_new[0]-size_xn, plane_origin_new[0]+size_xp)
    # # rangey = (plane_origin_new[1]-size_yn, plane_origin_new[1]+size_yp)
    # # rangez = (plane_origin_new[2]-size_zn, plane_origin_new[2]+size_zp)

    # # plane_points = [origin_point_physical, pointx_physical, pointy_physical]
    # # plane_origin = origin_point_physical
    # # volume_origin = origin_point

    # unit_vector_z_positive = plane_norm

    # grid_2d, corner_points = sampleGrid(sitk_image3d, plane_origin_new,
    #                                    unit_vector_x_positive,
    #                                    unit_vector_y_positive,
    #                                    unit_vector_z_positive,
    #                                    plane_size,
    #                                    spacing=spacing)

    # x0, y0, z0 = plane_origin_new[0], plane_origin_new[1], plane_origin_new[2]
    # d = a*x0 + b*y0 + c*z0

    # plane_params = [np.rad2deg(np.arccos(a)),
    #                 np.rad2deg(np.arccos(b)),
    #                 np.rad2deg(np.arccos(c)),
    #                 d]

    # return grid_2d, plane_norm, plane_origin_new, plane_params, corner_points


def calcMaxDistTwoPlanes(points1, points2):
    ''' distance metric between two planes.
    Returns maximum distance between two sets of points.
    '''
    return max(np.linalg.norm(np.array(points1)-np.array(points2), axis=1))

# def cropImage(sitk_image):

#     image_size = sitk_image.GetSize()
#     crop_size = [(int)(x/1.5) for x in image_size]
#     # crop_size = (128,128,10)
#     crop_index = [(int)(x/3) for x in crop_size]
#     # consider all z indexes
#     crop_size[2] = image_size[2]
#     crop_index[2] = 0

#     return sitk.RegionOfInterest(sitk_image,
#                                  size=crop_size,
#                                  index=crop_index)




###############################################################################
################################ main code ####################################
###############################################################################
if __name__ == "__main__":


    # directory = '/vol/medic01/users/aa16914/projects/tensorpack-medical/examples/PlaneDetection/data/cardiac/'
    directory = '/vol/medic01/users/aa16914/projects/tensorpack-medical/examples/PlaneDetection/data/cardiac/test/'

    # save_dir = '/vol/medic01/users/aa16914/projects/tensorpack-medical/examples/LandmarkDetection3D/ultrasound_fetal_brain_DQN/data/plane_detection/train/'

    train_files = trainFiles_cardio_plane(directory)
    sampled_files = train_files.sample_circular()

    plane_size = (200,200,9)
    spacing = (1,1,1)

    for i in range(train_files.num_files):

        # if i<44: continue

        sitk_image, sitk_image_2ch, sitk_image_4ch, filepath = next(sampled_files)
        filename = filepath.split('/')[-1]
        print(i, '-', filename)

        direction = np.array(sitk_image_4ch.GetDirection())
        transformation = np.array(direction.reshape(3,3))
        det = np.linalg.det(transformation)
        # origin_ = np.array(sitk_image_4ch.GetOrigin())
        # origin_ = -origin_
        # sitk_image_4ch.SetOrigin(origin_)

        # sitk_image_4ch = sitk.Flip(sitk_image_4ch,flipAxes=[True,False,False])

        # sitk_image = resampleIsotropic(sitk_image, spacing_new=(0.5,0.5,0.5))

        # # crop original image
        # sitk_image_cropped = cropImage(sitk_image)

        if np.sign(sitk_image_4ch.GetOrigin()[2])>0:
            sitk_image_4ch = sitk.Flip(sitk_image_4ch,flipAxes=[False,True,True])

        save_filename = directory + '4CH_rreg_fix_orient/' + filename +'.nii.gz'
        sitk.WriteImage(sitk_image_4ch, save_filename)



        # # sitk.WriteImage(sitk_image, 'sitk_image.nii.gz')
        # # sitk.WriteImage(sitk_image_4ch, 'sitk_image_4ch.nii.gz')
        # # sitk.WriteImage(sitk_image_2ch, 'sitk_image_2ch.nii.gz')
        # # sitk.WriteImage(sitk_image_cropped, 'sitk_image_cropped.nii.gz')

        # # create the figure
        # plt.close()
        # fig = plt.figure()

        # size_image3d = sitk_image.GetSize()
        # origin_point = [int(i/2) for i in size_image3d]

        # #######################################################################
        # # 2ch images
        # #######################################################################
        # # show the reference image
        # ax1 = fig.add_subplot(231)
        # ax1.text(30,30,det,fontsize=20,fontweight='bold',color='green')
        # # read images
        # data_2ch_gt = sitk.GetArrayFromImage(sitk_image_4ch)[0,:,:].T
        # ax1.imshow(data_2ch_gt, cmap=cm.Greys_r, interpolation='nearest')#, origin='lower', extent=[0,1,0,1])
        # plt.axis('off');
        # plt.title('');

        # # # Extract image_2ch image from vol3d
        # # plane_2ch_params = get2DPlaneFrom3DImageGivenTwoNiftis(
        # #                                         sitk_image3d=sitk_image,
        # #                                         sitk_image2d=sitk_image_2ch)
        # # # Extract image_2ch image from vol3d
        # # grid, corner_points = extract2DPlaneFrom3DImageGiven2DImage(
        # #                                         sitk_image3d=sitk_image,
        # #                                         sitk_image2d=sitk_image_2ch,
        # #                                         plane_size=plane_size)
        # # ax2 = fig.add_subplot(232)
        # # plt.imshow(grid[:,:,5], cmap=cm.Greys_r)#, interpolation='nearest', origin='lower', extent=[0,1,0,1],zorder=1)
        # # x_center, y_center, z_center = ndimage.measurements.center_of_mass(grid>0)
        # # plt.scatter(x_center,y_center, color='red', cmap= cm.CMRmap_r, linewidth='3')
        # # plt.axis('off');

        # # Extract image_2ch image from vol3d
        # plane_gt = Plane(*getGroundTruthPlane(sitk_image,
        #                                    sitk_image_4ch,
        #                                    origin= origin_point,
        #                                    plane_size=plane_size,
        #                                    spacing=spacing))
        # ax2 = fig.add_subplot(232)
        # plt.imshow(plane_gt.grid[:,:,5], cmap=cm.Greys_r)#, interpolation='nearest', origin='lower', extent=[0,1,0,1],zorder=1)
        # x_center, y_center, z_center = (plane_gt.grid.shape[0]/2,
        #                                 plane_gt.grid.shape[1]/2,
        #                                 plane_gt.grid.shape[2]/2)
        # plt.scatter(x_center,y_center, color='red', cmap= cm.CMRmap_r, linewidth='3')
        # plt.axis('off');

        # # Get initial plane
        # plane = Plane(*getInitialPlane(sitk_image3d=sitk_image,
        #                                plane_size=plane_size,
        #                                origin=origin_point,
        #                                spacing=spacing))
        # ax3 = fig.add_subplot(233);
        # plt.imshow(plane.grid[:,:,5], cmap=cm.Greys_r)
        # x_center, y_center, z_center = (plane.grid.shape[0]/2,
        #                                 plane.grid.shape[1]/2,
        #                                 plane.grid.shape[2]/2)
        # plt.axis('off');

        # #######################################################################
        # # 4ch images
        # #######################################################################
        # # print('origin origin ', sitk_image.GetOrigin())
        # # print('origin before ', sitk_image_4ch.GetOrigin())
        # # if np.sign(sitk_image_4ch.GetOrigin()[2])>0:
        # #     sitk_image_4ch = sitk.Flip(sitk_image_4ch,flipAxes=[False,True,True])
        # #     save_filename = directory + '4CH_rreg_fix_orient/' + filename +'.nii.gz'
        # #     sitk.WriteImage(sitk_image_4ch, save_filename)
        # # print('origin after ', sitk_image_4ch.GetOrigin())


        # direction = np.array(sitk_image_4ch.GetDirection())
        # transformation = np.array(direction.reshape(3,3))
        # det = np.linalg.det(transformation)

        # # show the reference image
        # ax4 = fig.add_subplot(234)
        # # read images
        # data_4ch_gt = sitk.GetArrayFromImage(sitk_image_4ch)[0,:,:].T
        # plt.imshow(data_4ch_gt, cmap=cm.Greys_r, interpolation='nearest')
        # ax4.text(30,30,det,fontsize=20,fontweight='bold',color='green')

        # #, origin='lower', extent=[0,1,0,1])
        # plt.axis('off');
        # plt.title('');
        # # # Extract image_3ch image from vol3d
        # # plane_4ch_params = get2DPlaneFrom3DImageGivenTwoNiftis(
        # #                                         sitk_image3d=sitk_image,
        # #                                         sitk_image2d=sitk_image_4ch)
        # # # Extract image_3ch image from vol3d
        # # grid, corner_points = extract2DPlaneFrom3DImageGiven2DImage(
        # #                                         sitk_image3d=sitk_image,
        # #                                         sitk_image2d=sitk_image_4ch,
        # #                                         plane_size=plane_size)
        # # ax6 = fig.add_subplot(234)
        # # plt.imshow(grid[:,:,5], cmap=cm.Greys_r)#, interpolation='nearest', origin='lower', extent=[0,1,0,1],zorder=1)
        # # x_center, y_center, z_center = ndimage.measurements.center_of_mass(grid>0)
        # # plt.scatter(x_center,y_center, color='red', cmap= cm.CMRmap_r, linewidth='3')
        # # plt.axis('off');

        # # Extract image_3ch image from vol3d
        # plane_gt = Plane(*getGroundTruthPlane(sitk_image,
        #                                    sitk_image_4ch,
        #                                    origin= origin_point,
        #                                    plane_size=plane_size,
        #                                    spacing=spacing))
        # ax5 = fig.add_subplot(235)
        # plt.imshow(plane_gt.grid[:,:,5], cmap=cm.Greys_r)#, interpolation='nearest', origin='lower', extent=[0,1,0,1],zorder=1)
        # x_center, y_center, z_center = (plane_gt.grid.shape[0]/2,
        #                                 plane_gt.grid.shape[1]/2,
        #                                 plane_gt.grid.shape[2]/2)
        # plt.scatter(x_center,y_center, color='red', cmap= cm.CMRmap_r, linewidth='3')
        # plt.axis('off');

        # # Get initial plane
        # plane = Plane(*getInitialPlane(sitk_image3d=sitk_image,
        #                                plane_size=plane_size,
        #                                origin=origin_point,
        #                                spacing=spacing))
        # ax6 = fig.add_subplot(236);
        # plt.imshow(plane.grid[:,:,5], cmap=cm.Greys_r)
        # x_center, y_center, z_center  = (plane.grid.shape[0]/2,
        #                                  plane.grid.shape[1]/2,
        #                                  plane.grid.shape[2]/2)
        # plt.scatter(x_center,y_center, color='red', cmap= cm.CMRmap_r, linewidth='3')
        # plt.axis('off');


        # # show full screen
        # mng = plt.get_current_fig_manager();
        # mng.resize(*mng.window.maxsize());


        # fig.show()
        # # time.sleep(10)

        # #######################################################################
        # # test playing
        # # play an action and get new plane
        # # plt.figure()
        # img = None
        # # plane_norm = plane_gt.norm
        # # plane_origin = plane.origin
        # plane.params[:-1] = deepcopy(plane_gt.params[:-1])
        # dist = np.inf
        # change_direction = False
        # while True:
        #     plane = Plane(*play(sitk_image,
        #                         origin_point,
        #                         plane.params,
        #                         plane_size = plane_size,
        #                         action = 1,
        #                         spacing = spacing,
        #                         change_direction = change_direction))

        #     dist_old = np.copy(dist)
        #     dist = calcMaxDistTwoPlanes(plane.points, plane_gt.points)
        #     # print('dist old = {} - dist = {}'.format(dist_old, dist))

        #     if (abs(dist) > abs(dist_old)):
        #         change_direction = True
        #         # print('change_direction', change_direction)

        #     # print('plane = {}, plane_gt = {}'.format(plane.params, plane_gt.params))

        #     # print('plane_gt.points', plane_gt.points)
        #     # print('plane.points', plane.points)
        #     # print('-------------------------------------------')

        #     im = plane.grid[:,:,5]

        #     if img is None:
        #         img = plt.imshow(im, cmap=cm.Greys_r)
        #         # plt.show();
        #     else:
        #         img.set_data(im)
        #         # plt.pause(.1)
        #         img.axes.figure.canvas.draw()

        #     if dist<0.5:
        #         change_direction = False
        #         plt.waitforbuttonpress()
        #         print('-------------------------------------------')

        #         break


        # break



###############################################################################
######################### plot 2D planes in 3D ################################
###############################################################################




# # show the 3D rotated projection
# # create a for vertex mesh
# xx, yy = np.meshgrid(np.linspace(0,image_dims_3d[0]-1,image_dims_3d[0]),
#                      np.linspace(0,image_dims_3d[1]-1,image_dims_3d[1]))

# # create vertices for a rotated mesh (3D rotation matrix)
# X = xx
# Y = yy
# Z = (image_dims_3d[2]/2)*np.ones(X.shape)

# ax3 = fig.add_subplot(133, projection='3d')
# cset = ax3.contourf(X, Y, image_30, 100, zdir='z', offset=image_dims_3d[2]/2, cmap=cm.Greys_r)
# # plt.axis('off');
# # ax2.set_zlim((0.,1.))
# ax3.set_xlim(0,image_dims_3d[0]);
# ax3.set_ylim(0,image_dims_3d[1]);
# ax3.set_zlim(0,image_dims_3d[2]);

# plt.colorbar(cset)

# mng = plt.get_current_fig_manager();
# mng.resize(*mng.window.maxsize());




