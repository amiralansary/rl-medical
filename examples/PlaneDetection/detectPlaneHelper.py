from IPython.core.debugger import set_trace
import SimpleITK as sitk
import numpy as np


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
    origin_grid = np.multiply.outer(np.ones(plane_size),origin_point)
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
    corner_points = [grid_final[0,0,0,:].tolist(),
                     grid_final[-1,0,0,:].tolist(),
                     grid_final[0,-1,0,:].tolist(),
                     grid_final[0,0,-1,:].tolist(),
                     grid_final[-1,-1,-1,:].tolist()]

    return grid, sorted(corner_points) #sorted(corner_points)



###############################################################################
###############################################################################

def getInitialPlane(sitk_image3d, plane_size,
                    origin_point=None, spacing=(1,1,1)):
    ''' This function extracts initial plane, which is the mid xy-plane
        Returns:
            plane in the norm form
    '''

    # extract three points in sitk_image3d coordinates
    if not origin_point:
        origin_point = [int(i/2) for i in sitk_image3d.GetSize()]

    pointx = (origin_point[0] + plane_size[0]/2,
              origin_point[1],
              origin_point[2])
    pointy = (origin_point[0],
              origin_point[1] + plane_size[1]/2,
              origin_point[2])
    pointz = (origin_point[0],
              origin_point[1],
              origin_point[2]+1)
    # physical coordinates
    plane_origin = sitk_image3d.TransformContinuousIndexToPhysicalPoint(origin_point)
    pointx_physical = sitk_image3d.TransformContinuousIndexToPhysicalPoint(
                        pointx)
    pointy_physical = sitk_image3d.TransformContinuousIndexToPhysicalPoint(
                        pointy)
    pointz_physical = sitk_image3d.TransformContinuousIndexToPhysicalPoint(
                        pointz)
    # find the correspondant plane in physical space
    # new plane d = -(ax + by + cz)
    unit_vector_x_positive = np.array(pointx_physical) - np.array(plane_origin)
    unit_vector_y_positive = np.array(pointy_physical) - np.array(plane_origin)
    unit_vector_z_positive = np.array(pointz_physical) - np.array(plane_origin)
    # normalise vectors
    unit_vector_x_positive = normalizeUnitVector(unit_vector_x_positive)
    unit_vector_y_positive = normalizeUnitVector(unit_vector_y_positive)
    unit_vector_z_positive = normalizeUnitVector(unit_vector_z_positive)

    # sample a grid of size[plane_size]
    grid_2d, corner_points = sampleGrid(sitk_image3d,plane_origin,
                                       unit_vector_x_positive,
                                       unit_vector_y_positive,
                                       unit_vector_z_positive,
                                       plane_size,
                                       spacing=spacing)

    # plane equation ax+by+cz=d , where norm = (a,b,c), and d=a*x0+b*y0+c*z0
    plane_norm = unit_vector_z_positive
    a, b, c = plane_norm[0], plane_norm[1], plane_norm[2]
    x0, y0, z0 = plane_origin[0], plane_origin[1], plane_origin[2]

    d = a*x0 + b*y0 + c*z0

    plane_params = [np.rad2deg(np.arccos(a)),
                    np.rad2deg(np.arccos(b)),
                    np.rad2deg(np.arccos(c)),
                    d]

    return grid_2d, plane_norm, plane_origin, plane_params, sorted(corner_points)


###############################################################################
###############################################################################

def getGroundTruthPlane(sitk_image3d,sitk_image2d,
                        origin_image3d,plane_size,spacing=(1,1,1)):
    ''' This function extracts the ground truth plane
        Returns:
            plane in the norm form
            corner points of the plane
    '''
    size_image2d = sitk_image2d.GetSize()
    size_image3d = sitk_image3d.GetSize()

    origin_image3d_physical = np.array(sitk_image3d.TransformContinuousIndexToPhysicalPoint(origin_image3d))

    # extract three points in sitk_image2d coordinates
    center_point2d = [int(i/2) for i in size_image2d]

    pointx2d = (center_point2d[0]+plane_size[0]/2,
                center_point2d[1],
                center_point2d[2])

    pointy2d = (center_point2d[0],
                center_point2d[1]+plane_size[1]/2,
                center_point2d[2])
    # transform these points to physical coordinates
    center_point2d_physical = sitk_image2d.TransformContinuousIndexToPhysicalPoint(center_point2d)
    pointx2d_physical = sitk_image2d.TransformContinuousIndexToPhysicalPoint(
                        pointx2d)
    pointy2d_physical = sitk_image2d.TransformContinuousIndexToPhysicalPoint(
                        pointy2d)
    # find the correspondant plane in physical space
    # new plane d = -(ax + by + cz)
    v1 = np.array(pointx2d_physical) - np.array(center_point2d_physical)
    v2 = np.array(pointy2d_physical) - np.array(center_point2d_physical)
    plane_norm = np.array(np.cross(v1, v2))
    # -------------------------------------------------------------------------
    # here the plane is defined using a plane norm and origin point
    # normalise norm vector
    plane_norm = normalizeUnitVector(plane_norm)
    # plane origin is defined by projecting the 3d origin on the plane
    plane_origin = projectPointOnPlane(origin_image3d_physical,
                                       plane_norm,
                                       center_point2d_physical)
    # -------------------------------------------------------------------------
    # find point in x-direction of the 3d volume to sample in this direction
    # sample a 2d grid of size[x,y] in xy-directions of the image_3d
    pointx3d = (origin_image3d[0] + plane_size[0]/2,
                origin_image3d[1],
                origin_image3d[2])
    # physical coordinates
    pointx3d_physical = sitk_image3d.TransformContinuousIndexToPhysicalPoint(
                        pointx3d)
    pointx3d_projected = projectPointOnPlane(pointx3d_physical,
                                             plane_norm,
                                             center_point2d_physical)
    # new plane coordinate unit vectors
    unit_vector_x_positive =[pointx3d_projected[0] - plane_origin[0],
                            pointx3d_projected[1] - plane_origin[1],
                            pointx3d_projected[2] - plane_origin[2]]
    unit_vector_y_positive = np.cross(plane_norm, unit_vector_x_positive)
    # normalize unit vectors
    unit_vector_x_positive = normalizeUnitVector(unit_vector_x_positive)
    unit_vector_y_positive = normalizeUnitVector(unit_vector_y_positive)
    unit_vector_z_positive = np.copy(plane_norm)

    grid_2d, corner_points = sampleGrid(sitk_image3d,plane_origin,
                                       unit_vector_x_positive,
                                       unit_vector_y_positive,
                                       unit_vector_z_positive,
                                       plane_size,
                                       spacing=spacing)
    # plane equation ax+by+cz=d , where norm = (a,b,c), and d=a*x0+b*y0+c*z0
    a, b, c = plane_norm[0], plane_norm[1], plane_norm[2]
    x0, y0, z0 = plane_origin[0], plane_origin[1], plane_origin[2]

    d = a*x0 + b*y0 + c*z0

    plane_params = [np.rad2deg(np.arccos(a)),
                    np.rad2deg(np.arccos(b)),
                    np.rad2deg(np.arccos(c)),
                    d]

    return grid_2d, plane_norm, plane_origin, plane_params, sorted(corner_points)

###############################################################################

def getPlane(sitk_image3d, origin_image3d,
        plane_params, plane_size, spacing=(1,1,1)):
    ''' Get a plane from a 3d nifti image using its norm form
    '''
    # plane equation ax+by+cz=d , where norm = (a,b,c), and d=a*x0+b*y0+c*z0
    a, b, c, d = [np.cos(np.deg2rad(plane_params[0])),
                  np.cos(np.deg2rad(plane_params[1])),
                  np.cos(np.deg2rad(plane_params[2])),
                  plane_params[3]]
    # print('a {}, b {}, c {}, d {}'.format(a, b, c, d))

    # plane_norm = normalizeUnitVector(np.array((a,b,c)))
    plane_norm = np.array((a,b,c))

    # get point on the new plane
    origin_image3d_physical = sitk_image3d.TransformContinuousIndexToPhysicalPoint(origin_image3d)
    x, y = origin_image3d_physical[0:2]
    z = (d - a*x - b*y) / c
    point_plane_new = (x,y,z)
    # project the 3d origin on the new plane to get the new plane origin
    plane_origin = projectPointOnPlane(origin_image3d_physical, plane_norm, point_plane_new)
    # select a random point on the x-axis of image3d
    pointx3d = (origin_image3d[0] + plane_size[0]/2,
                origin_image3d[1],
                origin_image3d[2])
    pointx3d_physical = sitk_image3d.TransformContinuousIndexToPhysicalPoint(pointx3d)
    pointx_projected = projectPointOnPlane(pointx3d_physical,
                                           plane_norm, plane_origin)
    # unit vectors
    unit_vector_x_positive = np.array((pointx_projected[0] - plane_origin[0],
                                    pointx_projected[1] - plane_origin[1],
                                    pointx_projected[2] - plane_origin[2]))
    # find the y-axis
    unit_vector_y_positive = np.cross(plane_norm, unit_vector_x_positive)
    # normalize unit vectors
    unit_vector_x_positive = normalizeUnitVector(unit_vector_x_positive)
    unit_vector_y_positive = normalizeUnitVector(unit_vector_y_positive)
    # copy the z-axis from plane norm
    unit_vector_z_positive = np.copy(plane_norm)
    # unit_vector_z_positive = normalizeUnitVector(unit_vector_z_positive)
    # get the grid and corner points
    grid_2d, corner_points = sampleGrid(sitk_image3d, plane_origin,
                                       unit_vector_x_positive,
                                       unit_vector_y_positive,
                                       unit_vector_z_positive,
                                       plane_size,
                                       spacing=spacing)

    # # plane equation ax+by+cz=d , where norm = (a,b,c), and d=a*x0+b*y0+c*z0
    # a, b, c = plane_norm[0], plane_norm[1], plane_norm[2]
    # x0, y0, z0 = plane_origin[0], plane_origin[1], plane_origin[2]

    # d = a*x0 + b*y0 + c*z0

    # plane_params = [np.rad2deg(np.arccos(a)),
    #                 np.rad2deg(np.arccos(b)),
    #                 np.rad2deg(np.arccos(c)),
    #                 d]

    return grid_2d, plane_norm, plane_origin, plane_params, sorted(corner_points)


###############################################################################
def calcMaxDistTwoPlanes(points1, points2):
    ''' distance metric between two planes.
    Returns maximum distance between two sets of points.
    '''
    return max(np.linalg.norm(np.array(points1)-np.array(points2), axis=1))

def calcDistTwoParams(params1, params2):
    ''' distance metric between two parameters of planes.
    '''
    return np.linalg.norm(abs(np.array(params1)-np.array(params2)))


def projectPointOnPlane(point, norm, point_in_plane):
    # project the origin point in 3d on the plane
    # The distance from origin_3d to the plane, is simply the length of the
    # projection of a vector between this point and any poing ton the plane
    # onto the unit normal vector.
    # Since norm is length one, this distance is simply the absolute value of
    # the dot product d = v⋅n. Finally add d to the 3d origin point
    v = np.array(point_in_plane) - point
    d_plane = norm.dot(v)
    return point + d_plane*norm

def normalizeUnitVector(vector):
    ''' return the normalized unit vector '''
    return np.array(vector) / np.linalg.norm(vector)

def checkOriginLocation(sitk_image, origin_point):
    ''' check if the origin point is inside the 3d image dimensions
        Returns
            Boolean: true if the point is outside
    '''
    point_location = sitk_image.TransformPhysicalPointToIndex(origin_point)
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
    check1 = [abs(i-j)>18000 for i,j in zip(params1[:-1],params2[:-1])]
    # the diff between translation within [-100,100] mm
    check2 = abs(params1[-1] - params2[-1]) > 150
    go_out = np.logical_or(sum(check1), check2)

    return go_out
