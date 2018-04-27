#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: detectPlanePlayer.py
# Author: Amir Alansary <amiralansary@gmail.com>


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.simplefilter("ignore", category=PendingDeprecationWarning)


import os
import six
import random
import threading
import numpy as np
from tensorpack import logger
from collections import (Counter, defaultdict, deque, namedtuple)

from collections import deque
import copy

import cv2
import math
import time
from PIL import Image
import subprocess
import shutil

import gym
from gym import spaces

try:
    import pyglet
except ImportError as e:
    reraise(suffix="HINT: you can install pyglet directly via 'pip install pyglet'.")

from tensorpack.utils.utils import get_rng
from tensorpack.utils.stats import StatCounter

from sampleTrain import *
from detectPlaneHelper import *


import csv


__all__ = ['MedicalPlayer']

_ALE_LOCK = threading.Lock()


# plane container of its array, normal vector, origin,
# and parameters(angles in degrees and d), selected points (e.g. corners)
Plane = namedtuple('Plane', ['grid', 'grid_smooth', 'norm', 'origin', 'params',
                            'points'])

# ===================================================================
# =================== 3d medical environment ========================
# ===================================================================

from IPython.core.debugger import set_trace
# set_trace()

class MedicalPlayer(gym.Env):
    """Class that provides 3D medical image environment.
    This is just an implementation of the classic "agent-environment loop".
    Each time-step, the agent chooses an action, and the environment returns
    an observation and a reward."""

    def __init__(self, directory=None, files_list=None, viz=False, train=False,
                 screen_dims=(27,27,27), spacing=(1,1,1), nullop_start=30,
                 history_length=30, max_num_frames=0, saveGif=False,
                 saveVideo=False):
        """
        :param train_directory: environment or game name
        :param viz: visualization
            set to 0 to disable
            set to +ve number to be the delay between frames to show
            set to a string to be the directory for storing frames
        :param screen_dims: shape of the frame cropped from the image to feed
            it to dqn (d,w,h) - defaults (27,27,27)
        :param nullop_start: start with random number of null ops
        :param history_length: consider lost of lives as end of
            episode (useful for training)
        """
        super(MedicalPlayer, self).__init__()

        self.reset_stat()
        self._supervised = False
        self._init_action_angle_step = 8
        self._init_action_dist_step = 4


        ## save results in csv file
        # brain_adult_spacing_multi_3_actionh_8_4_unsupervised_duel_double_smooth_batch_32_layers_8_fold_1_model_2000000
        self.csvfile = 'dummy.csv'
        if not train:
            with open(self.csvfile, 'w') as outcsv:
                fields = ["filename", "dist_error", "angle_error"]
                writer = csv.writer(outcsv)
                writer.writerow(map(lambda x: x, fields))

        # read files from directory - add your data loader here
        self.files = filesListBrainMRPlane(directory,files_list)

        # prepare file sampler
        self.sampled_files = self.files.sample_circular()
        self.filepath = None
        # maximum number of frames (steps) per episodes
        self.cnt = 0
        self.max_num_frames = max_num_frames
        # stores information: terminal, score, distError
        self.info = None
        # option to save display as gif
        self.saveGif = saveGif
        self.saveVideo = saveVideo
        # training flag
        self.train = train
        # image dimension (2D/3D)
        self._plane_size = screen_dims
        self.dims = len(self._plane_size)
        if self.dims == 2:
            self.width, self.height = self._plane_size
        else:
            self.width, self.height, self.depth = self._plane_size
        # plane sampling spacings
        self.init_spacing = np.array(spacing)
        # stat counter to store current score or accumlated reward
        self.current_episode_score = StatCounter()
        # get action space and minimal action set
        self.action_space = spaces.Discrete(8) # change number actions here
        self.actions = self.action_space.n
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=self._plane_size)
        # history buffer for storing last locations to check oscillations
        self._history_length = history_length
        # circular buffer to store plane parameters history [4,history_length]
        self._plane_history = deque(maxlen=self._history_length)
        self._bestq_history = deque(maxlen=self._history_length)
        self._dist_history = deque(maxlen=self._history_length)
        self._dist_history_params = deque(maxlen=self._history_length)
        self._dist_supervised_history = deque(maxlen=self._history_length)
        # self._loc_history = [(0,) * self.dims] * self._history_length
        self._loc_history = [(0,) * 4] * self._history_length
        self._qvalues_history = [(0,) * self.actions] * self._history_length
        self._qvalues = [0,] * self.actions

        with _ALE_LOCK:
            self.rng = get_rng(self)
            # visualization setup
            if isinstance(viz, six.string_types):   # check if viz is a string
                assert os.path.isdir(viz), viz
                viz = 0
            if isinstance(viz, int):
                viz = float(viz)
            self.viz = viz
            if self.viz and isinstance(self.viz, float):
                self.viewer = None
                self.gif_buffer = []

        self._restart_episode()

    # -------------------------------------------------------------------------

    def _reset(self):
        # with _ALE_LOCK:
        self._restart_episode()
        return self._current_state()

    def _restart_episode(self):
        """
        restart current episoide
        """
        self.terminal = False
        self.cnt = 0 # counter to limit number of steps per episodes
        self.num_games.feed(1)
        self.current_episode_score.reset()  # reset score stat counter
        self._plane_history.clear()
        self._bestq_history.clear()
        self._dist_history.clear()
        self._dist_history_params.clear()
        self._dist_supervised_history.clear()
        # self._loc_history = [(0,) * self.dims] * self._history_length
        self._loc_history = [(0,) * 4] * self._history_length
        self._qvalues_history = [(0,) * self.actions] * self._history_length

        self.new_random_game()

    # -------------------------------------------------------------------------

    def new_random_game(self):
        # print('\n============== new game ===============\n')
        self.terminal = False
        self.viewer = None

        # sample a new image
        (self.sitk_image, self.landmarks, self.filepath) = next(self.sampled_files)
        self.filename = os.path.basename(self.filepath)
        # image volume size
        self._image_dims = self.sitk_image.GetSize()
        self.action_angle_step = copy.deepcopy(self._init_action_angle_step)
        self.action_dist_step = copy.deepcopy(self._init_action_dist_step)
        self.spacing = self.init_spacing.copy()

        # ---------------------------------------------------------------------
        # Extract landmarks and Get ground truth plane
        # logger.info('filename {} '.format(self.filename))
        self.ac_point = np.round(self.landmarks[13]).astype('int')
        self.pc_point = np.round(self.landmarks[14]).astype('int')
        self.midsag_point = np.round(self.landmarks[2]).astype('int')
        # fix point axes.astype('int')
        shared_x = np.round(np.mean(self.landmarks[:9,0])).astype('int')
        shared_z = np.round(np.mean(self.landmarks[13:15,2])).astype('int')
        self.ac_point[0] = shared_x
        self.pc_point[0] = shared_x
        self.ac_point[2] = shared_z
        self.pc_point[2] = shared_z
        self.midsag_point[0] = shared_x
        # self._origin3d_point = deepcopy(self.pc_point)
        self._origin3d_point = np.array([int(i/2) for i in self._image_dims])
        self._groundTruth_plane = Plane(*getACPCPlaneFromLandmarks(
                                        self.sitk_image,
                                        self._origin3d_point.astype('float'),
                                        self.ac_point, self.pc_point,
                                        self.midsag_point,
                                        self._plane_size, self.spacing))
        # get an istropic 1mm groundtruth plane
        image_size = (int(min(self._image_dims)),)*3
        self.groundTruth_plane_iso = Plane(*getACPCPlaneFromLandmarks(
                                            self.sitk_image,
                                            self._origin3d_point.astype('float'),
                                            self.ac_point, self.pc_point,
                                            self.midsag_point,
                                            image_size, [1,1,1]))
        self.landmarks_gt = self.landmarks[13:15]
        # self._groundTruth_plane = Plane(*getMidSagPlaneFromLandmarks(
        #                                 self.sitk_image,
        #                                 self._origin3d_point.astype('float'),
        #                                 self.ac_point, self.pc_point,
        #                                 self.midsag_point,
        #                                 self._plane_size, self.spacing))
        # self.landmarks_gt = self.landmarks[0:3]

        # logger.info('groundTruth {}'.format(self._groundTruth_plane.params))

        # ---------------------------------------------------------------------
        # Get a random initial plane and set current plane the same
        x = self.rng.randint(0,self._image_dims[0])
        y = self.rng.randint(0,self._image_dims[1])
        z = self.rng.randint(0,self._image_dims[2])

        # fix starting plane for evaluation miccai paper
        if not self.train: x = y = z = 10

        point = (int(x), int(y), int(z))
        norm = normalizeUnitVector(point - self._origin3d_point)
        thetax = np.rad2deg(np.arccos(norm[0]))
        thetay = np.rad2deg(np.arccos(norm[1]))
        thetaz = np.rad2deg(np.arccos(norm[2]))
        init_plane_params = (thetax,thetay,thetaz,0)

        self._plane = self._init_plane = Plane(*getPlane(
                                            self.sitk_image,
                                            self._origin3d_point,
                                            init_plane_params,
                                            self._plane_size,
                                            self.spacing))
        # calculate current distance between initial and ground truth planes
        # self.cur_dist = calcMeanDistTwoPlanes(self._groundTruth_plane.points,
        #                                      self._plane.points)
        self.cur_dist_params = calcDistTwoParams(self._groundTruth_plane.params
                                        , self._plane.params,
                                        scale_angle = self.action_angle_step,
                                        scale_dist = self.action_dist_step)

        self._screen = self._current_state()
        _, dist = zip(*[projectPointOnPlane(point, self._plane.norm, self._plane.origin) for point in self.landmarks_gt])
        self.cur_dist = np.mean(np.abs(dist))

    # -------------------------------------------------------------------------

    def step(self, act, qvalues):
        """The environment's step function returns exactly what we need.
        Args:
          action:
        Returns:
          observation (object):
            an environment-specific object representing your observation of the environment. For example, pixel data from a camera, joint angles and joint velocities of a robot, or the board state in a board game.
          reward (float):
            amount of reward achieved by the previous action. The scale varies between environments, but the goal is always to increase your total reward.
          done (boolean):
            whether it's time to reset the environment again. Most (but not all) tasks are divided up into well-defined episodes, and done being True indicates the episode has terminated. (For example, perhaps the pole tipped too far, or you lost your last life.)
          info (dict):
            diagnostic information useful for debugging. It can sometimes be useful for learning (for example, it might contain the raw probabilities behind the environment's last state change). However, official evaluations of your agent are not allowed to use this for learning.
        """
        self.terminal = False
        self._qvalues = qvalues
        # get current plane params
        current_plane_params = np.copy(self._plane.params)
        next_plane_params = current_plane_params.copy()
        # ---------------------------------------------------------------------
        # theta x+ (param a)
        if (act==0): next_plane_params[0] += self.action_angle_step
        # theta y+ (param b)
        if (act==1): next_plane_params[1] += self.action_angle_step
        # theta z+ (param c)
        if (act==2): next_plane_params[2] += self.action_angle_step
        # dist d+
        if (act==3): next_plane_params[3] += self.action_dist_step

        # theta x- (param a)
        if (act==4): next_plane_params[0] -= self.action_angle_step
        # theta y- (param b)
        if (act==5): next_plane_params[1] -= self.action_angle_step
        # theta z- (param c)
        if (act==6): next_plane_params[2] -= self.action_angle_step
        # dist d-
        if (act==7): next_plane_params[3] -= self.action_dist_step
        # ---------------------------------------------------------------------
        # self.reward = self._calc_reward_points(self._plane.points,
        #                                        next_plane.points)
        self.reward = self._calc_reward_params(current_plane_params,
                                               next_plane_params)
        # threshold reward between -1 and 1
        self.reward = np.sign(np.around(self.reward,decimals=1))
        go_out = False

        if self._supervised and self.train:
            ## supervised
            dist_queue = deque(maxlen=self.actions)
            plane_params_queue = deque(maxlen=self.actions)
            # theta x+ (param a)
            next_plane_params = np.copy(self._plane.params)
            next_plane_params[0] += self.action_angle_step
            plane_params_queue.append(next_plane_params)
            # dist_queue.append(calcMeanDistTwoPlanes(self._groundTruth_plane.points, plane_params_queue[-1].points))
            dist_queue.append(calcDistTwoParams(
                                        self._groundTruth_plane.params,
                                        plane_params_queue[-1],
                                        scale_angle = self.action_angle_step,
                                        scale_dist = self.action_dist_step))
            # theta y+ (param b) ----------------------------------------------
            next_plane_params = np.copy(self._plane.params)
            next_plane_params[1] += self.action_angle_step
            plane_params_queue.append(next_plane_params)
            # dist_queue.append(calcMeanDistTwoPlanes(self._groundTruth_plane.points, plane_params_queue[-1].points))
            dist_queue.append(calcDistTwoParams(
                                        self._groundTruth_plane.params,
                                        plane_params_queue[-1],
                                        scale_angle = self.action_angle_step,
                                        scale_dist = self.action_dist_step))
            # theta z+ (param c) ----------------------------------------------
            next_plane_params = np.copy(self._plane.params)
            next_plane_params[2] += self.action_angle_step
            plane_params_queue.append(next_plane_params)
            # dist_queue.append(calcMeanDistTwoPlanes(self._groundTruth_plane.points, plane_params_queue[-1].points))
            dist_queue.append(calcDistTwoParams(
                                        self._groundTruth_plane.params,
                                        plane_params_queue[-1],
                                        scale_angle = self.action_angle_step,
                                        scale_dist = self.action_dist_step))
            # dist d+ ---------------------------------------------------------
            next_plane_params = np.copy(self._plane.params)
            next_plane_params[3] += self.action_dist_step
            plane_params_queue.append(next_plane_params)
            # dist_queue.append(calcMeanDistTwoPlanes(self._groundTruth_plane.points, plane_params_queue[-1].points))
            dist_queue.append(calcDistTwoParams(
                                        self._groundTruth_plane.params,
                                        plane_params_queue[-1],
                                        scale_angle = self.action_angle_step,
                                        scale_dist = self.action_dist_step))
            # theta x- (param a) ----------------------------------------------
            next_plane_params = np.copy(self._plane.params)
            next_plane_params[0] -= self.action_angle_step
            plane_params_queue.append(next_plane_params)
            # dist_queue.append(calcMeanDistTwoPlanes(self._groundTruth_plane.points, plane_params_queue[-1].points))
            dist_queue.append(calcDistTwoParams(
                                        self._groundTruth_plane.params,
                                        plane_params_queue[-1],
                                        scale_angle = self.action_angle_step,
                                        scale_dist = self.action_dist_step))
            # theta y- (param b) ----------------------------------------------
            next_plane_params = np.copy(self._plane.params)
            next_plane_params[1] -= self.action_angle_step
            plane_params_queue.append(next_plane_params)
            # dist_queue.append(calcMeanDistTwoPlanes(self._groundTruth_plane.points, plane_params_queue[-1].points))
            dist_queue.append(calcDistTwoParams(
                                        self._groundTruth_plane.params,
                                        plane_params_queue[-1],
                                        scale_angle = self.action_angle_step,
                                        scale_dist = self.action_dist_step))
            # theta z- (param c) ----------------------------------------------
            next_plane_params = np.copy(self._plane.params)
            next_plane_params[2] -= self.action_angle_step
            plane_params_queue.append(next_plane_params)
            # dist_queue.append(calcMeanDistTwoPlanes(self._groundTruth_plane.points, plane_params_queue[-1].points))
            dist_queue.append(calcDistTwoParams(
                                        self._groundTruth_plane.params,
                                        plane_params_queue[-1],
                                        scale_angle = self.action_angle_step,
                                        scale_dist = self.action_dist_step))
            # dist d- ---------------------------------------------------------
            next_plane_params = np.copy(self._plane.params)
            next_plane_params[3] -= self.action_dist_step
            plane_params_queue.append(next_plane_params)
            # dist_queue.append(calcMeanDistTwoPlanes(self._groundTruth_plane.points, plane_params_queue[-1].points))
            dist_queue.append(calcDistTwoParams(
                                        self._groundTruth_plane.params,
                                        plane_params_queue[-1],
                                        scale_angle = self.action_angle_step,
                                        scale_dist = self.action_dist_step))

            # -----------------------------------------------------------------
            # get best plane based on lowest distance to the target
            next_plane_idx = np.argmin(dist_queue)
            next_plane = Plane(*getPlane(self.sitk_image,
                                        self._origin3d_point,
                                        plane_params_queue[next_plane_idx],
                                        self._plane_size,
                                        spacing=self.spacing))
            self._dist_supervised_history.append(np.min(dist_queue))

        else:
            ## unsupervised or testing
            # get the new plane using new params result from taking the action
            next_plane = Plane(*getPlane(self.sitk_image,
                                        self._origin3d_point,
                                        next_plane_params,
                                        self._plane_size,
                                        spacing=self.spacing))
            # -----------------------------------------------------------------
            # check if the screen is not full of zeros (background)
            go_out = checkBackgroundRatio(next_plane,
                                          min_pixel_val=0.5, ratio=0.8)
            # also check if go out (sampling from outside the volume)
            # by checking if the new origin
            if not go_out:
                go_out = checkOriginLocation(self.sitk_image,next_plane.origin)
            # also check if plane parameters got very high
            if not go_out:
                go_out = checkParamsBound(next_plane.params,
                                          self._groundTruth_plane.params)
            # punish lowest reward if the agent tries to go out and keep same plane
            if go_out:
                self.reward = -1 # lowest possible reward
                next_plane = copy.deepcopy(self._plane)
                if self.train: self.terminal = True # end episode and restart

        # ---------------------------------------------------------------------
        # update current plane
        self._plane = copy.deepcopy(next_plane)
        # terminate if maximum number of steps is reached
        self.cnt += 1
        if self.cnt >= self.max_num_frames: self.terminal = True

        # check oscillation and reduce action step or terminate if minimum
        if self._oscillate:
            if self.train and self._supervised:
                self._plane = self.getBestPlaneTrain()
            else:
                self._plane = self.getBestPlane()

            # find distance metrics
            # self.cur_dist = calcMeanDistTwoPlanes(self._groundTruth_plane.points, self._plane.points)
            _, dist = zip(*[projectPointOnPlane(point, self._plane.norm, self._plane.origin) for point in self.landmarks_gt])
            self.cur_dist = np.mean(np.abs(dist))
            self._update_heirarchical()
            self._clear_history()
            # terminate if distance steps are less than 1
            if self.action_dist_step < 1:   self.terminal = True

        # ---------------------------------------------------------------------
        # find distance error
        _, dist = zip(*[projectPointOnPlane(point, self._plane.norm, self._plane.origin) for point in self.landmarks_gt])
        self.cur_dist = np.mean(np.abs(dist))
        self.cur_dist_params = calcDistTwoParams(self._groundTruth_plane.params                            , self._plane.params,
                                        scale_angle = self.action_angle_step,
                                        scale_dist = self.action_dist_step)
        self.current_episode_score.feed(self.reward)
        self._update_history() # store results in memory

        # terminate if distance between params are low during training
        if self.train and (self.cur_dist_params<=1):
            self.terminal = True
            self.num_success.feed(1)

        # ---------------------------------------------------------------------

        # # supervised reward (for debuging)
        # reward_supervised = self._calc_reward_params(current_plane_params,
        #                                        self._plane.params)
        # # threshold reward between -1 and 1
        # self.reward = np.sign(np.around(reward_supervised,decimals=1))

        # ---------------------------------------------------------------------

        # render screen if viz is on
        if self.viz:
            if isinstance(self.viz, float):
                self.display()

        A = normalizeUnitVector(self._groundTruth_plane.norm)
        B = normalizeUnitVector(self._plane.norm)
        angle_between_norms = np.rad2deg(np.arccos(A.dot(B)))

        info = {'score': self.current_episode_score.sum, 'gameOver': self.terminal,
                'distError': self.cur_dist, 'distAngle': angle_between_norms,
                'filename':self.filename}

        if self.terminal:
            with open(self.csvfile, 'a') as outcsv:
                fields= [self.filename, self.cur_dist, angle_between_norms]
                writer = csv.writer(outcsv)
                writer.writerow(map(lambda x: x, fields))

        return self._current_state(), self.reward, self.terminal, info



    # -------------------------------------------------------------------------

    def _update_heirarchical(self):
        self.action_angle_step = int(self.action_angle_step/2)
        self.action_dist_step = self.action_dist_step-1
        if (self.spacing[0] > 1): self.spacing -= 1
        self._groundTruth_plane = Plane(*getACPCPlaneFromLandmarks(
                                        self.sitk_image,
                                        self._origin3d_point.astype('float'),
                                        self.ac_point, self.pc_point,
                                        self.midsag_point,
                                        self._plane_size, self.spacing))
        # self._groundTruth_plane = Plane(*getMidSagPlaneFromLandmarks(
        #                                 self.sitk_image,
        #                                 self._origin3d_point.astype('float'),
        #                                 self.ac_point, self.pc_point,
        #                                 self.midsag_point,
        #                                 self._plane_size, self.spacing))
        # logger.info('update hierarchical - spacing = {} - angle step = {} - dist step = {}'.format(self.spacing,self.action_angle_step,self.action_dist_step))


    def getBestPlane(self):
        ''' get best location with best qvalue from last for locations
        stored in history
        '''
        best_idx = np.argmin(self._bestq_history)
        # best_idx = np.argmax(self._bestq_history)
        return self._plane_history[best_idx]

    def getBestPlaneTrain(self):
        ''' get best location with best qvalue from last for locations
        stored in history
        '''
        best_idx = np.argmin(self._dist_supervised_history)
        # best_idx = np.argmax(self._bestq_history)
        return self._plane_history[best_idx]

    def _current_state(self):
        """
        :returns: a gray-scale (h, w, d) float ###uint8 image
        """
        return self._plane.grid_smooth


    def _clear_history(self):
        self._plane_history.clear()
        self._bestq_history.clear()
        self._dist_history.clear()
        self._dist_history_params.clear()
        self._dist_supervised_history.clear()
        # self._loc_history = [(0,) * self.dims] * self._history_length
        self._loc_history = [(0,) * 4] * self._history_length
        self._qvalues_history = [(0,) * self.actions] * self._history_length



    def _update_history(self):
        ''' update history buffer with current state
        '''
        # update location history
        self._loc_history[:-1] = self._loc_history[1:]
        # loc = self._plane.origin
        loc = self._plane.params
        # logger.info('loc {}'.format(loc))
        self._loc_history[-1] = (np.around(loc[0],decimals=2),
                                 np.around(loc[1],decimals=2),
                                 np.around(loc[2],decimals=2),
                                 np.around(loc[3],decimals=2))
        # update distance history
        self._dist_history.append(self.cur_dist)
        self._dist_history_params.append(self.cur_dist_params)
        # update params history
        self._plane_history.append(self._plane)
        self._bestq_history.append(np.max(self._qvalues))
        # update q-value history
        self._qvalues_history[:-1] = self._qvalues_history[1:]
        self._qvalues_history[-1] = self._qvalues

    def _calc_reward_points(self, prev_points, next_points):
        ''' Calculate the new reward based on the euclidean distance to the target plane
        '''
        prev_dist = calcMeanDistTwoPlanes(self._groundTruth_plane.points,
                                         prev_points)
        next_dist = calcMeanDistTwoPlanes(self._groundTruth_plane.points,
                                         next_points)

        return prev_dist - next_dist

    def _calc_reward_params(self, prev_params, next_params):
        ''' Calculate the new reward based on the euclidean distance to the target plane
        '''
        # logger.info('prev_params {}'.format(np.around(prev_params,2)))
        # logger.info('next_params {}'.format(np.around(next_params,2)))
        prev_dist = calcScaledDistTwoParams(self._groundTruth_plane.params,
                                      prev_params,
                                      scale_angle = self.action_angle_step,
                                      scale_dist = self.action_dist_step)
        next_dist = calcScaledDistTwoParams(self._groundTruth_plane.params,
                                      next_params,
                                      scale_angle = self.action_angle_step,
                                      scale_dist = self.action_dist_step)
        # logger.info('next_dist {} prev_dist {}'.format(next_dist, prev_dist))
        return prev_dist - next_dist

    @property
    def _oscillate(self):
        ''' Return True if the agent is stuck and oscillating
        '''
        counter = Counter(self._loc_history)
        freq = counter.most_common()
        # return false is history is empty (begining of the game)
        if len(freq) < 2: return False
        # check frequency
        if freq[0][0] == (0,0,0,0):
            if (freq[1][1]>3):
                # logger.info('oscillating {}'.format(self._loc_history))
                return True
            else:
                return False
        elif (freq[0][1]>3):
            # logger.info('oscillating {}'.format(self._loc_history))
            return True

    def get_action_meanings(self):
        ''' return array of integers for actions '''
        ACTION_MEANING = {
            0 : "inc_x",    # increment +1 the norm angle in x-direction
            1 : "inc_y",    # increment +1 the norm angle in y-direction
            2 : "inc_z",    # increment +1 the norm angle in z-direction
            3 : "inc_d",    # increment +1 the norm distance d to origin
            4 : "dec_x",    # decrement -1 the norm angle in x-direction
            5 : "dec_y",    # decrement -1 the norm angle in y-direction
            6 : "dec_z",    # decrement -1 the norm angle in z-direction
            7 : "dec_d",    # decrement -1 the norm distance d to origin
        }
        return [ACTION_MEANING[i] for i in self.actions]

    @property
    def getScreenDims(self):
        """
        return screen dimensions
        """
        return (self.width, self.height, self.depth)

    def lives(self):
        return None

    def reset_stat(self):
        """ Reset all statistics counter"""
        self.stats = defaultdict(list)
        self.num_games = StatCounter()
        self.num_success = StatCounter()


    def display(self, return_rgb_array=False):
        # pass
        # --------------------------------------------------------------------
        ## planes seen by the agent
        # # get image and convert it to pyglet
        # plane = self._plane.grid[:,:,round(self.depth/2)] # z-plane
        # # concatenate groundtruth image
        # gt_plane = self._groundTruth_plane.grid[:,:,round(self.depth/2)]
        # --------------------------------------------------------------------
        ## whole plan
        image_size = (int(min(self._image_dims)),)*3
        current_plane = Plane(*getPlane(self.sitk_image,
                                self._origin3d_point,
                                self._plane.params,
                                image_size,
                                spacing=[1,1,1]))

        # get image and convert it to pyglet
        plane = current_plane.grid[:,:,int(image_size[2]/2)] # z-plane
        # concatenate groundtruth image
        gt_plane = self.groundTruth_plane_iso.grid[:,:,int(image_size[2]/2)]
        # --------------------------------------------------------------------
        # concatenate two planes side by side
        plane = np.concatenate((plane,gt_plane),axis=1)
        #
        img = cv2.cvtColor(plane,cv2.COLOR_GRAY2RGB) # congvert to rgb
        # rescale image
        # INTER_NEAREST, INTER_LINEAR, INTER_AREA, INTER_CUBIC, INTER_LANCZOS4
        scale_x = 5
        scale_y = 5
        # img = cv2.resize(img,
        #                  (int(scale_x*img.shape[1]),int(scale_y*img.shape[0])),
        #                  interpolation=cv2.INTER_LINEAR)
        # skip if there is a viewer open
        if (not self.viewer) and self.viz:
            from viewer import SimpleImageViewer
            self.viewer = SimpleImageViewer(arr=img,
                                            scale_x=1,
                                            scale_y=1,
                                            filepath=self.filename)
            self.gif_buffer = []

        # display image
        self.viewer.draw_image(img)
        self.viewer.display_text('Current Plane', color=(0,0,204,255),
                                 x=int(0.7*img.shape[1]/7), y=img.shape[0]-3)
        self.viewer.display_text('Ground Truth', color=(0,0,204,255),
                                 x=int(4.3*img.shape[1]/7), y=img.shape[0]-3)

        # display info
        dist_color_flag = False
        if len(self._dist_history)>1:
            dist_color_flag = self.cur_dist<self._dist_history[-2]

        color_dist = (0,204,0,255) if dist_color_flag else (204,0,0,255)
        text = 'Error ' + str(round(self.cur_dist,3)) + 'mm'
        self.viewer.display_text(text, color=color_dist,
                                 x=int(3*img.shape[1]/8), y=5*scale_y)

        dist_color_flag = False
        if len(self._dist_history_params)>1:
            dist_color_flag = self.cur_dist_params<self._dist_history_params[-2]

        # color_dist = (0,255,0,255) if dist_color_flag else (255,0,0,255)
        # text = 'Params Error ' + str(round(self.cur_dist_params,3))
        # self.viewer.display_text(text, color=color_dist,
                                 # x=int(6*img.shape[1]/8), y=5*scale_y)
        text = 'Spacing ' + str(round(self.spacing[0],3)) + 'mm'
        self.viewer.display_text(text, color=(204,204,0,255),
                                 x=int(6*img.shape[1]/8), y=5*scale_y)

        color_reward = (0,204,0,255) if self.reward>0 else (204,0,0,255)
        text = 'Reward ' + "%+d" % round(self.reward,3)
        self.viewer.display_text(text, color=color_reward,
                                 x=2*scale_x, y=5*scale_y)

        # render and wait (viz) time between frames
        self.viewer.render()

        # save gif
        if self.saveGif:
            image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
            data = image_data.get_data('RGB', image_data.width * 3)
            # set_trace()
            arr = np.array(bytearray(data)).astype('uint8')
            arr = np.flip(np.reshape(arr,(image_data.height, image_data.width, -1)),0)
            im = Image.fromarray(arr).convert('P')
            self.gif_buffer.append(im)

            if not self.terminal:
                gifname = self.filename.split('.')[0] + '.gif'
                self.viewer.savegif(gifname,arr=self.gif_buffer, duration=self.viz)
        if self.saveVideo:
            dirname = 'tmp_video'
            if (self.cnt <=1):
                if os.path.isdir(dirname):
                    logger.warn("""Log directory {} exists! Use 'd' to delete it. """.format(dirname))
                    act = input("select action: d (delete) / q (quit): ").lower().strip()
                    if act == 'd':
                        shutil.rmtree(dirname, ignore_errors=True)
                    else:
                        raise OSError("Directory {} exits!".format(dirname))
                os.mkdir(dirname)

            frame = dirname + '/' + '%04d' % self.cnt + '.png'
            pyglet.image.get_buffer_manager().get_color_buffer().save(frame)
            if self.terminal:
                save_cmd = ['ffmpeg','-f', 'image2', '-framerate', '30',
                    '-pattern_type', 'sequence', '-start_number', '0', '-r',
                    '3', '-i', dirname + '/%04d.png', '-s', '1280x720',
                    '-vcodec', 'libx264', '-b:v', '2567k', self.filename+'.mp4']
                subprocess.check_output(save_cmd)
                shutil.rmtree(dirname, ignore_errors=True)


class DiscreteActionSpace(object):

    def __init__(self, num):
        super(DiscreteActionSpace, self).__init__()
        self.num = num
        self.rng = get_rng(self)

    def sample(self):
        return self.rng.randint(self.num)

    def num_actions(self):
        return self.num

    def __repr__(self):
        return "DiscreteActionSpace({})".format(self.num)

    def __str__(self):
        return "DiscreteActionSpace({})".format(self.num)

# =============================================================================
# ================================ FrameStack =================================
# =============================================================================
class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Buffer observations and stack across channels (last axis)."""
        gym.Wrapper.__init__(self, env)
        self.k = k # history length
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        # chan = 1 if len(shp) == 2 else shp[2]
        self._base_dim = len(shp)
        new_shape = shp + (k,)
        # self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], chan * k))
        self.observation_space = spaces.Box(low=0, high=255, shape=new_shape)

    def _reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        ob = self.env.reset()
        for _ in range(self.k - 1):
            self.frames.append(np.zeros_like(ob))
        self.frames.append(ob)
        return self._observation()

    def step(self, action, qvalues):
        ob, reward, done, info = self.env.step(action,qvalues)
        self.frames.append(ob)
        return self._observation(), reward, done, info

    def _observation(self):
        assert len(self.frames) == self.k
        return np.stack(self.frames, axis=-1)
        # if self._base_dim == 2:
        #     return np.stack(self.frames, axis=-1)
        # else:
        #     return np.concatenate(self.frames, axis=2)


# =============================================================================
# ================================== notes ====================================
# =============================================================================
"""

## Notes from landmark detection Siemens paper
# states  -> ROI - center current pos - size (2D 60x60) (3D 26x26x26)
# actions -> move (up, down, left, right)
# rewards -> delta(d) relative distance change after executing a move (action)

# re-sample -> isotropic (2D 2mm) (3D 1mm)

# gamma = 0.9 , replay memory size P = 100000 , learning rate = 0.00025
# net : 3 conv+pool - 3 FC+dropout (3D kernels for 3d data)

# navigate till oscillation happen (terminate when infinite loop)

# location is a high-confidence landmark -> if the expected reward from this location is max(q*(s_target,a))<1 the agent is closer than one pixel

# object is not in the image: oscillation occurs at points where max(q)>4


## Other Notes:

    DeepMind's original DQN paper
        used frame skipping (for fast playing/learning) and
        applied pixel-wise max to consecutive frames (to handle flickering).

    so an input to the neural network is consisted of four frame;
        [max(T-1, T), max(T+3, T+4), max(T+7, T+8), max(T+11, T+12)]

    ALE provides mechanism for frame skipping (combined with adjustable random action repeat) and color averaging over skipped frames. This is also used in simple_dqn's ALEEnvironment

    Gym's Atari Environment has built-in stochastic frame skipping common to all games. So the frames returned from environment are not consecutive.

    The reason behind Gym's stochastic frame skipping is, as mentioned above, to make environment stochastic. (I guess without this, the game will be completely deterministic?)
    cf. in original DQN and simple_dqn same randomness is achieved by having agent performs random number of dummy actions at the beginning of each episode.

    I think if you want to reproduce the behavior of the original DQN paper, the easiest will be disabling frame skip and color averaging in ALEEnvironment then construct the mechanism on agent side.


"""
