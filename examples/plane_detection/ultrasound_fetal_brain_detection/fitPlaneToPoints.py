
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.optimize
import functools
from math import (cos,sin,radians)

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

def cross(a, b):
    ''' returns the cross product between two vectors a and b
    '''
    return [a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]]

def rotate_plane(angle,origin,points):
    # convert angle from degree to radians
    a,b,c = [radians(i) for i in angle]
    # center of rotation
    x0,y0,z0 = origin

    # rotation matrix http://www.songho.ca/opengl/files/gl_anglestoaxes05.png
    T = np.array([

                 [cos(b)*cos(c),
                 (sin(a)*sin(b)*cos(c) + cos(a)*sin(c)),
                 (sin(a)*sin(c) - cos(a)*sin(b)*cos(c)),
                 0],

                 [-cos(b)*sin(c),
                 (cos(a)*cos(c) - sin(a)*sin(b)*sin(c)),
                 (sin(a)*cos(c) + cos(a)*sin(b)*sin(c)),
                 0],

                 [sin(b),
                 -sin(a)*cos(b),
                 cos(a)*cos(b),
                 0],

                 [0, 0, 0, 1]

                 ])

    print(T)

    unit_vector = np.ones(points[0].size)
    old_points = np.array([
                          points[0].ravel()-x0,
                          points[1].ravel()-y0,
                          points[2].ravel()-z0,
                          unit_vector
                          ])

    new_points = T.dot(old_points)

    xx = np.resize(new_points[0:]+x0,(2,2))
    yy = np.resize(new_points[1:]+y0,(2,2))
    zz = np.resize(new_points[2:]+z0,(2,2))

    return xx,yy,zz




###############################################################################
############################# read image ######################################
###############################################################################

import SimpleITK as sitk

filename = '/vol/medic01/users/aa16914/projects/tensorpack-medical/examples/LandmarkDetection3D/ultrasound_fetal_brain_DQN/data/train_align/images/ifind52-98.nii.gz'


sitk_image = sitk.ReadImage(filename, sitk.sitkFloat32)
image = sitk.GetArrayFromImage(sitk_image)
image_spacing = sitk_image.GetSpacing()
image_dims = image.shape

xx, yy = np.meshgrid([0,image_dims[0]], [0,image_dims[1]])



###############################################################################
########################## fit landmarks plane ################################
###############################################################################

points_gt = [(158.8458824368, 108.9053266703, 144.0943445134),
             (81.5498490344, 113.2481069306, 134.9776846378),
             (245.3943647194, 106.4349331587, 152.1697652106),
             (161.5142826747, 173.1474228028, 142.0315437855),
             (157.8218610222, 46.5594213910, 145.3059673435),
             (135.8380636409, 111.0182991436, 141.6557054305),
             (108.6811780185, 111.4425963193, 137.8388745404)]

# optimization function - the error between gt_points and fitted plane
opt_func = functools.partial(error, points=points_gt)
# initialize plane equation parameters z = a*x + b*y + c with zeros
plane_initial_params = [0, 0, 0]
# final parameter result values
res = scipy.optimize.minimize(opt_func, plane_initial_params)

a_final = res.x[0]
b_final = res.x[1]
c_final = res.x[2]


###############################################################################
############################ plot points and plane ############################
###############################################################################

# unpack points
xs_gt, ys_gt, zs_gt = zip(*points_gt)
# prepare figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# plot scatter points
ax.scatter(xs_gt, ys_gt, zs_gt)
# test few more points
ax.scatter(100,0,100*a_final+c_final, color='red');
ax.scatter(0,100,100*b_final+c_final, color='green');
ax.scatter(0,0,c_final, color='yellow');


###############################################################################
######################## find plane directional angles ########################
###############################################################################
# find the normal vector using the cross product between two unit vectors of
# in the direction of x and y of the new plane
v1 = np.array([1,0,a_final+c_final]) - np.array([0,0,c_final])
v2 = np.array([0,1,b_final+c_final]) - np.array([0,0,c_final])

v_norm_fit_plane = np.array(np.cross(v1, v2))
print('v_norm_fit_plane {}'.format(v_norm_fit_plane))
# normalise norm vector on the fitted plane
v_norm_fit_plane = v_norm_fit_plane * 1.0 / np.linalg.norm(v_norm_fit_plane)
print('v_norm_fit_plane {}'.format(v_norm_fit_plane))

# pick the origin point of the new plane at (0,0,c) equivalant to (0,0,1)
origin_point_gt = np.array([0.0, 0.0, c_final])
# find the dot product between the norm and origin point sitting on the
# new plane d = -(ax + by + cz)
d_gt = -origin_point_gt.dot(v_norm_fit_plane)

# Point-normal form and general form of the equation of a plane
# ax + by + cz + d = 0 , where (a,b,c) is the normal vector
zz_gt = (-v_norm_fit_plane[0] * xx - v_norm_fit_plane[1] * yy - d_gt) * 1. /v_norm_fit_plane[2]

print('zz_gt {}'.format(zz_gt))

zz_gt = xx * a_final + yy * b_final + c_final
print('zz_gt {}'.format(zz_gt))

# plot the new plane surface
ax.plot_surface(xx, yy, zz_gt, alpha=0.5, color=[0,1,0])

vlength = 100*np.linalg.norm(v_norm_fit_plane)
plt.quiver(0,0,c_final,
           v_norm_fit_plane[0], v_norm_fit_plane[1], v_norm_fit_plane[2],
           pivot='tail',length=vlength,arrow_length_ratio=0.2/vlength,
           color='black')
plt.quiver(0,0,c_final,
           1,0,a_final,
           pivot='tail',length=vlength,arrow_length_ratio=0.2/vlength,
           color='blue')
plt.quiver(0,0,c_final,
           0,1,b_final,
           pivot='tail',length=vlength,arrow_length_ratio=0.2/vlength,
           color='red')
plt.quiver(0,0,c_final,
           0,0,1,
           pivot='tail',length=vlength,arrow_length_ratio=0.2/vlength,
           color='green')



















## A simple example to check if norm calculations are correct
# p1 = np.array((161.5142826747, 173.1474228028, 142.0315437855))
# p2 = np.array((157.8218610222, 46.5594213910, 145.3059673435))
# p3 = np.array((135.8380636409, 111.0182991436, 141.6557054305))

# v1 = p2 - p1
# v2 = p3 - p1

# v_norm_fit_plane = np.array(np.cross(v1, v2))
# print('v_norm_fit_plane {}'.format(v_norm_fit_plane))
# # normalise norm vector on the fitted plane
# v_norm_fit_plane = v_norm_fit_plane * 1.0 / np.linalg.norm(v_norm_fit_plane)
# print('v_norm_fit_plane {}'.format(v_norm_fit_plane))


# vlength = 100*np.linalg.norm(v_norm_fit_plane)
# plt.quiver(p1[0],p1[1],p1[2],
#            v_norm_fit_plane[0], v_norm_fit_plane[1], v_norm_fit_plane[2],
#            pivot='tail',length=vlength,#arrow_length_ratio=0.2/vlength,
#            color='black')

# calculate anglbe between new plane and horizontal
nomral_horizontal = np.array([0,0,1])
cos_angle = nomral_horizontal.dot(v_norm_fit_plane)
angle = np.rad2deg(np.arccos(cos_angle / (np.linalg.norm(nomral_horizontal) *
                                          np.linalg.norm(v_norm_fit_plane))))

v3 = v_norm_fit_plane - [0,0,c_final]

anglex = np.rad2deg(np.arctan2(np.linalg.norm(np.multiply(v3,[1,0,0])),
                            v3.dot([1,0,0]))) #- 90
angley = np.rad2deg(np.arctan2(np.linalg.norm(np.multiply(v3,[0,1,0])),
                            v3.dot([0,1,0]))) #- 90
anglez = np.rad2deg(np.arctan2(np.linalg.norm(np.multiply(v3,[0,0,1])),
                            v3.dot([0,0,1]))) #- 90


# v = np.array([0,0,1])

# anglex = np.rad2deg(np.arctan2(np.linalg.norm(np.multiply(v,[1,0,0])),
#                             v.dot([1,0,0]))) #- 90
# angley = np.rad2deg(np.arctan2(np.linalg.norm(np.multiply(v,[0,1,0])),
#                             v.dot([0,1,0]))) #- 90
# anglez = np.rad2deg(np.arctan2(np.linalg.norm(np.multiply(v,[0,0,1])),
#                             v.dot([0,0,1]))) #- 90


print('angle {}'.format(angle))








###############################################################################
############################## find cross lines ###############################
###############################################################################

# first line
# (0. topHead, 1. centerHead, 2. frontHead, 3. backHead, 12. CSP, 13. midpointCSP)
# p0 = np.array((0.0000000000, 0.0000000000, 0.0000000000))
p1 = np.array((158.8458824368, 108.9053266703, 144.0943445134))
p2 = np.array((81.5498490344, 113.2481069306, 134.9776846378))
p3 = np.array((245.3943647194, 106.4349331587, 152.1697652106))
p12 = np.array((135.8380636409, 111.0182991436, 141.6557054305))
p13 = np.array((108.6811780185, 111.4425963193, 137.8388745404))

points1 = np.array((p1,p2,p3,p12,p13))

# second line
# (1. centerHead, 4. rightHead, 5. leftHead)
p4 = np.array((161.5142826747, 173.1474228028, 142.0315437855))
p5 = np.array((157.8218610222, 46.5594213910, 145.3059673435))

points2 = np.array((p1,p4,p5))

def fit_line(data):
    # Calculate the mean of the points, i.e. the 'center' of the cloud
    datamean = data.mean(axis=0)

    # Do an SVD on the mean-centered data.
    uu, dd, vv = np.linalg.svd(data - datamean)

    # Now vv[0] contains the first principal component, i.e. the direction
    # vector of the 'best fit' line in the least squares sense.

    # Now generate some points along this best fit line, for plotting.

    # I use -7, 7 since the spread of the data is roughly 14
    # and we want it to have mean 0 (like the points we did
    # the svd on). Also, it's a straight line, so we only need 2 points.
    linepts = vv[0] * np.mgrid[-100:100:2j][:, np.newaxis]

    # shift by the mean to get the line in the right place
    linepts += datamean

    return linepts


# fit line to points
linepts1 = fit_line(points1)
linepts2 = fit_line(points2)
ax.plot(*linepts1.T)
ax.plot(*linepts2.T)



###############################################################################
###############################################################################

# project the csp on the fitted plane
csp_orig = (135.8380636409, 111.0182991436, 141.6557054305)
zz_projected = (-v_norm_fit_plane[0] * 135.8380636409 - v_norm_fit_plane[1] * 111.0182991436 - d_gt) * 1. /v_norm_fit_plane[2]
csp_projected = (135.8380636409, 111.0182991436, zz_projected)

# ax.scatter(csp_projected[0], csp_projected[1], csp_projected[2], color='black')

center = np.array(image_dims)/2
# rotate
xx1, yy1, zz1 = rotate_plane(angle=(anglex,angley,anglez),
                             origin=[center[0],center[1],center[0]*a_final+center[1]*b_final+c_final],#csp_projected,
                             points=[xx,yy,zz_gt])

# plot the new plane surface
# ax.plot_surface(xx1, yy1, zz1, alpha=0.5, color=[0,0,1])



###############################################################################
ax.set_xlim(0,image_dims[0])
ax.set_ylim(0,image_dims[1])
ax.set_zlim(0,image_dims[2])

plt.show()



###############################################################################
########################### rotate sitk volume ################################
###############################################################################
import  multiprocessing
num_cores = multiprocessing.cpu_count()



def rotate_sitk(fixed_image,
                translation=(0,0,0),
                rotation_angle=(0,0,0),
                rotation_center=(0,0,0),
                interpolator=sitk.sitkBSpline,
                spacing=None):

    # get transformation parameters
    theta_x, theta_y, theta_z = (np.deg2rad(i) for i in rotation_angle)

    rigid_euler = sitk.Euler3DTransform(rotation_center, theta_x, theta_y, theta_z, translation)

    similarity = sitk.Similarity3DTransform()
    similarity.SetMatrix(rigid_euler.GetMatrix())
    similarity.SetTranslation(rigid_euler.GetTranslation())
    similarity.SetCenter(rigid_euler.GetCenter())


    # affine = sitk.AffineTransform(3)
    # affine.SetMatrix(similarity.GetMatrix())
    # affine.SetTranslation(similarity.GetTranslation())
    # affine.SetCenter(similarity.GetCenter())

    # get sampling parameters
    size = fixed_image.GetSize()
    origin = fixed_image.GetOrigin()
    direction = fixed_image.GetDirection()
    minMaxFilter = sitk.MinimumMaximumImageFilter()
    minMaxFilter.Execute(fixed_image)
    DefaultPixelValue = (float)(minMaxFilter.GetMinimum())

    if spacing is None:
        spacing = fixed_image.GetSpacing()

    # resample filter
    resampleFilter = sitk.ResampleImageFilter()
    resampleFilter.SetTransform(similarity)
    resampleFilter.SetOutputDirection(direction)
    resampleFilter.SetInterpolator(interpolator)
    resampleFilter.SetOutputSpacing(spacing)
    resampleFilter.SetOutputOrigin(origin)
    resampleFilter.SetDefaultPixelValue(DefaultPixelValue)
    resampleFilter.SetSize(size)
    resampleFilter.SetNumberOfThreads(num_cores)


    # transform the image
    moving_image = resampleFilter.Execute(fixed_image)

    return moving_image


# rotation_center = sitk_image.TransformContinuousIndexToPhysicalPoint(csp_projected)
# sitk_image_rotated = rotate_sitk(sitk_image,rotation_angle=(0,0,90), rotation_center=rotation_center)
# sitk.WriteImage(sitk_image_rotated,'image_rotated.nii.gz')

# size_x, size_y, size_z = sitk_image_rotated.GetSize()

# # slice new rotated image
# start = (0,0,(int)(csp_projected[2]-4))
# end = (size_x,size_y,(int)(csp_projected[2]+4))
# new_slice = sitk.Slice(sitk_image_rotated,start,end)
# sitk.WriteImage(new_slice, 'new_slice.nii.gz')

# # slice new rotated image
# size_x, size_y, size_z = sitk_image.GetSize()
# start = (0,0,(int)(csp_orig[2]-4))
# end = (size_x,size_y,(int)(csp_orig[2]+4))
# old_slice = sitk.Slice(sitk_image,start,end)
# sitk.WriteImage(old_slice, 'old_slice.nii.gz')

###############################################################################
############################ random CSP plane #################################
###############################################################################
# import random

# # extract a random plane containg two random points and CSP
# random_point_1 = (random.randint(10,image_dims[0]-10),
#                   random.randint(10,image_dims[1]-10),
#                   random.randint(10,image_dims[2]-10))

# random_point_2 = (random.randint(10,image_dims[0]-10),
#                   random.randint(10,image_dims[1]-10),
#                   random.randint(10,image_dims[2    ]-10))

# points_new = [csp_projected, random_point_1, random_point_2]

# opt_func = functools.partial(error, points=points_new)
# plane_initial_params = [0, 0, 0]
# res = scipy.optimize.minimize(opt_func, plane_initial_params)

# a = res.x[0]
# b = res.x[1]
# c = res.x[2]

# # unpack points
# xs, ys, zs = zip(*points_new)


# # plot scatter points
# ax.scatter(xs, ys, zs, color='black')

# # find the normal vector using the cross product between two unit vectors of
# # in the direction of x and y of the new plane
# normal = np.array(cross([1,0,a], [0,1,b]))
# # pick the origin point of the new plane at (0,0,c) equivalant to (0,0,1)
# point  = np.array([0.0, 0.0, c])
# # find the dot product between the norm and origin point sitting on the
# # new plane d = -(ax + by + cz)
# d = -point.dot(normal)

# # Point-normal form and general form of the equation of a plane
# # ax + by + cz + d = 0 , where (a,b,c) is the normal vector
# # more details on how to transoform between different plane forms
# # http://www.easy-math.net/transforming-between-plane-forms/
# zz = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]

# print(zz)

# # plot the new plane surface
# ax.plot_surface(xx, yy, zz, alpha=0.5, color=[1,0,0])

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
