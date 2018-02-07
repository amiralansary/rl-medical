from IPython.core.debugger import set_trace
from tensorpack import logger
from copy import deepcopy
import SimpleITK as sitk
import numpy as np

from scipy import ndimage

###############################################################################

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
    # smooth grid to get rid of noise resulting from sampling
    # grid = ndimage.uniform_filter(grid, size=3)

    # select few points
    points = [grid_final[0,0,0,:].tolist(),
              grid_final[-1,0,0,:].tolist(),
              grid_final[0,-1,0,:].tolist(),
              grid_final[0,0,-1,:].tolist(),
              grid_final[-1,-1,-1,:].tolist()]

    return grid, points #sorted(corner_points)



###############################################################################
###############################################################################

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


###############################################################################
###############################################################################

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

    # set_trace()
    # import matplotlib.pyplot as plt
    # plt.imshow(grid[:,:,5], cmap=plt.cm.Greys_r)
    # plt.show()
    # set_trace()

    return grid, plane_norm, plane_origin, plane_params, points

###############################################################################

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

    # logger.info('plane_norm {}'.format(np.around(plane_norm,2)))
    # logger.info('plane_origin {}'.format(np.around(plane_origin,2)))

    return grid, plane_norm, plane_origin, plane_params, points


###############################################################################
def calcMaxDistTwoPlanes(points1, points2):
    ''' distance metric between two planes.
    Returns maximum distance between two sets of points.
    '''
    return np.max(np.linalg.norm(np.array(points1)-np.array(points2), axis=1))

###############################################################################
def calcMeanDistTwoPlanes(points1, points2):
    ''' distance metric between two planes.
    Returns maximum distance between two sets of points.
    '''
    return np.mean(np.linalg.norm(np.array(points1)-np.array(points2), axis=1))

def calcDistTwoParams(params1, params2, scale_angle, scale_dist):
    ''' distance metric between two parameters of planes.
    '''
    params1 = np.array(params1)
    params2 = np.array(params2)

    # params1[:-1] /= scale_angle
    # params1[-1]  /= scale_dist

    # params2[:-1] /= scale_angle
    # params2[-1]  /= scale_dist

    dist = params1 - params2
    dist[:-1] /= scale_angle
    dist[-1] /= scale_dist

    # return np.linalg.norm(dist)
    return np.sum(abs(dist))


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

def normalizeUnitVector(vector):
    ''' return the normalized unit vector '''

    abs_vector = np.linalg.norm(vector)

    if (abs_vector==0):
        return np.array(vector)
    else:
        return np.array(vector) / np.linalg.norm(vector)

def checkOriginLocation(sitk_image, origin_point):
    ''' check if the origin point is inside the 3d image dimensions
        Returns
            Boolean: true if the point is outside
    '''
    # point_location = sitk_image.TransformPhysicalPointToIndex(origin_point)
    point_location = origin_point
    image_size = sitk_image.GetSize()
    # check if it is less than zero
    check1 = [i<=0 for i in point_location]
    # check if it is larger than image boundaries
    check2 = [i>=j for i, j in zip(point_location, image_size)]
    # combine both
    go_out = np.logical_or(sum(check1), sum(check2))

    return go_out


def checkBackgroundRatio(plane, min_pixel_val=0.5, ratio=0.8):
    ''' check ratio full of background is not larger than (default 80%)
        Returns
            Boolean: true if background pixels
    '''
    total = plane.grid.size
    # count non-zero pixels larger than (> 0.5)
    nonzero_count = np.count_nonzero(plane.grid>min_pixel_val)
    zero_ratio = (total-nonzero_count)/total

    return (zero_ratio > ratio)

def checkParamsBound(params1, params2):
    ''' bound the range between paramters '''
    # the diff between angles within [-180,180] degrees
    check1 = [abs(i-j)>180 for i,j in zip(params1[:-1],params2[:-1])]
    # the diff between translation within [-100,100] mm
    check2 = abs(params1[-1] - params2[-1]) > 100
    go_out = np.logical_or(sum(check1), check2)

    return go_out
