#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: detectPlanePlayer.py
# Author: Amir Alansary <amiralansary@gmail.com>

import os
import six
import random
import threading
import numpy as np
from tensorpack import logger
from collections import (Counter, defaultdict, deque, namedtuple)

import cv2
import math
import time
from PIL import Image

import gym
from gym import spaces

try:
    import pyglet
except ImportError as e:
    reraise(suffix="HINT: you can install pyglet directly via 'pip install pyglet'. But if you really just want to install all Gym dependencies and not have to think about it, 'pip install -e .[all]' or 'pip install gym[all]' will do it.")


from tensorpack.utils.utils import get_rng
from tensorpack.utils.stats import StatCounter
# from tensorpack.RL.envbase import RLEnvironment, DiscreteActionSpace


from viewer import SimpleImageViewer
from sampleTrain import *
from detectPlaneHelper import *


__all__ = ['MedicalPlayer']

_ALE_LOCK = threading.Lock()


# plane container of its array, normal vector, origin,
# and parameters(angles in degrees and d), selected points (e.g. corners)
Plane = namedtuple('Plane', ['grid', 'norm', 'origin', 'params', 'points'])

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

    def __init__(self, directory=None, viz=False, train=False,
                 screen_dims=(27,27,27), spacing=(1,1,1), nullop_start=30,
                 history_length=16, max_num_frames=0, savegif=False):
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

        # read files from directory - add your data loader here
        self.train_files = trainFiles_cardio_plane(directory)
        # self.train_files = trainFiles_fetal_US(directory)
        # self.train_files = trainFiles_cardio(directory)
        # self.train_files = trainFiles(directory)
        # prepare file sampler
        self.sampled_files = self.train_files.sample_circular()
        self.filepath = None
        # counter to limit number of steps per episodes
        self.cnt = 0
        # maximum number of frames (steps) per episodes
        self.max_num_frames = max_num_frames
        # stores information: terminal, score, distError
        self.info = None
        # option to save display as gif
        self.savegif = savegif
        # training flag
        self.train = train
        # image dimension (2D/3D)
        self.screen_dims = screen_dims
        self._plane_size = screen_dims
        self.dims = len(self.screen_dims)
        if self.dims == 2:
            self.width, self.height = screen_dims
        else:
            self.width, self.height, self.depth = screen_dims
        # plane sampling spacings
        self.spacing = spacing
        # history buffer for storing last locations to check oscilations
        self._history_length = history_length
        # circular buffer to store plane parameters history [4,history_length]
        self._params_history = list([(0,0,0,0)]) * self._history_length
        self._dist_history = [0] * self._history_length
        # stat counter to store current score or accumlated reward
        self.current_episode_score = StatCounter()
        # get action space and minimal action set
        self.action_space = spaces.Discrete(8) # change number actions here
        self.actions = self.action_space.n
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=self.screen_dims)
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
        self.num_games.feed(1)
        self.current_episode_score.reset()  # reset the stat counter
        self._params_history = list([(0,0,0,0)]) * self._history_length
        self._dist_history = [0] * self._history_length
        self.new_random_game()

    # -------------------------------------------------------------------------

    def new_random_game(self):
        # print('\n============== new game ===============\n')
        self.terminal = False
        self.viewer = None
        # sample a new image
        self.sitk_image, self.sitk_image_2ch, self.sitk_image_4ch, self.filepath = next(self.sampled_files)
        self.filename = os.path.basename(self.filepath)
        # image volume size
        self._image_dims = self.sitk_image.GetSize()

        # find center point of the initial plane
        if self.train:
            # sample randomly ±10% around the center point
            skip_thickness = ((int)(self._image_dims[0]/2.5),
                              (int)(self._image_dims[1]/2.5),
                              (int)(self._image_dims[2]/2.5))
            x = self.rng.randint(0+skip_thickness[0],
                             self._image_dims[0]-skip_thickness[0])
            y = self.rng.randint(0+skip_thickness[1],
                             self._image_dims[1]-skip_thickness[1])
            z = self.rng.randint(0+skip_thickness[2],
                             self._image_dims[2]-skip_thickness[2])
        else:
            # during testing start sample a plane around the center point
            x,y,z = ((int)(self._image_dims[0]/2),
                    (int)(self._image_dims[1]/2),
                    (int)(self._image_dims[2]/2))
        self._origin3d_point = (x,y,z)
        # Get ground truth plane
        self._groundTruth_plane = Plane(*getGroundTruthPlane(
                                            self.sitk_image,
                                            self.sitk_image_4ch,
                                            self._origin3d_point,
                                            self._plane_size,
                                            spacing=self.spacing))
        # Get initial plane and set current plane the same
        self._plane = self._init_plane = Plane(*getInitialPlane(
                                            sitk_image3d=self.sitk_image,
                                            plane_size=self._plane_size,
                                            origin_point=self._origin3d_point,
                                            spacing=self.spacing))
        # calculate current distance between initial and ground truth planes
        self.cur_dist = calcMaxDistTwoPlanes(self._groundTruth_plane.points,
                                             self._init_plane.points)
        self._screen = self._current_state()

    # -------------------------------------------------------------------------

    def _step(self, act):
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
        # get current plane params
        current_plane_params = self._plane.params
        next_plane_params = current_plane_params.copy()
        # ---------------------------------------------------------------------
        # theta x+ (param a)
        if (act==0): next_plane_params[0] += 1
        # theta y+ (param b)
        if (act==1): next_plane_params[1] += 1
        # theta z+ (param c)
        if (act==2): next_plane_params[2] += 1
        # dist d+
        if (act==3): next_plane_params[3] += 1

        # theta x- (param a)
        if (act==4): next_plane_params[0] -= 1
        # theta y- (param b)
        if (act==5): next_plane_params[1] -= 1
        # theta z- (param c)
        if (act==6): next_plane_params[2] -= 1
        # dist d-
        if (act==7): next_plane_params[3] -= 1
        # ---------------------------------------------------------------------
        # get the new plane using new params result from taking the action
        new_plane = Plane(*getPlane(self.sitk_image,
                                    self._origin3d_point,
                                    next_plane_params,
                                    self._plane_size,
                                    spacing=self.spacing))
        # check if the screen is not full of zeros (background)
        go_out = checkBackgroundRatio(new_plane, min_pixel_val=0.5, ratio=0.8)
        # also check if go out (sampling from outside the volume)
        # by checking if the new origin
        if not go_out:
            go_out = checkOriginLocation(self.sitk_image,new_plane.origin)
        # punish -1 reward if the agent tries to go out and keep same plane
        if go_out:
            self.reward = -1
            new_plane = self._plane
            # self.terminal = True # end episode and restart
        else:
            self.reward = self._calc_reward(self._plane.points, new_plane.points)

        # update screen, reward ,location, terminal
        self._plane = new_plane
        self._screen = self._current_state()
        self.cur_dist = calcMaxDistTwoPlanes(self._groundTruth_plane.points,
                                             self._plane.points)
        # store results in memory
        self._update_history()
        # termination conditon for train/test
        if self.train:
            if self.cur_dist<0.2:
                self.terminal = True
                self.num_success.feed(1)
        else:
            # check if agent oscillates
            if self._oscillate: self.terminal = True
            if self.cur_dist<1: self.num_success.feed(1)

        # render screen if viz is on
        with _ALE_LOCK:
            if self.viz:
                if isinstance(self.viz, float):
                    self.display()

        self.cnt += 1
        if self.cnt >= self.max_num_frames:
            self.terminal = True
            self.cnt = 0

        distance_error = self.cur_dist
        self.current_episode_score.feed(self.reward)

        info = {'score': self.current_episode_score.sum, 'gameOver': self.terminal, 'distError': distance_error}

        # this done in exp replay
        # if self.terminal:   self.reset()

        return self._current_state(), self.reward, self.terminal, info

    # -------------------------------------------------------------------------

    def _current_state(self):
        """
        :returns: a gray-scale (h, w, d) float ###uint8 image
        """
        return self._plane.grid

    def _update_history(self):
        ''' update history buffer with current state
        '''
        # update distance history
        self._dist_history[:-1] = self._dist_history[1:]
        self._dist_history[-1] = self.cur_dist
        # update params history
        self._params_history[:-1] = self._params_history[1:]
        self._params_history[-1] = self.cur_dist
        # TODO: update q-value history

    def _calc_reward(self, prev_points, next_points):
        ''' Calculate the new reward based on the euclidean distance to the target plane
        '''
        prev_dist = calcMaxDistTwoPlanes(self._groundTruth_plane.points,
                                         prev_points)
        next_dist = calcMaxDistTwoPlanes(self._groundTruth_plane.points,
                                         next_points)
        return prev_dist - next_dist

    @property
    def _oscillate(self):
        ''' Return True if the agent is stuck and oscillating
        '''
        counter = Counter(self._dist_history)
        freq = counter.most_common()

        if freq[0][0] == (0,0,0,0):
            return False
        elif (freq[0][1]>3):
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
        # get dimensions
        current_point = self._location
        target_point = self._target_loc
        # get image and convert it to pyglet
        plane = self.get_plane(current_point[2])# z-plane
        img = cv2.cvtColor(plane,cv2.COLOR_GRAY2RGB) # congvert to rgb
        # skip if there is a viewer open
        if self.viewer is None:
            self.viewer = SimpleImageViewer(arr=img,
                                            scale_x=2,
                                            scale_y=2,
                                            filepath=self.filename)
            self.gif_buffer = []
        # display image
        self.viewer.draw_image(img)
        # draw a transparent circle around target point with variable radius
        # based on the difference z-direction
        diff_z = abs(current_point[2]-target_point[2])
        self.viewer.draw_circle(radius=diff_z,
                                pos_x=target_point[0],
                                pos_y=target_point[1],
                                color=(1.0,0.0,0.0,0.2))
        # draw target point
        self.viewer.draw_circle(radius=1,
                                pos_x=target_point[0],
                                pos_y=target_point[1],
                                color=(1.0,0.0,0.0,1.0))
        # draw current point
        self.viewer.draw_circle(radius=1,
                                pos_x=current_point[0],
                                pos_y=current_point[1],
                                color=(0.0,0.0,1.0,1.0))

        # render and wait (viz) time between frames
        self.viewer.render()
        # time.sleep(self.viz)
        # save gif
        if self.savegif:
            image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
            data = image_data.get_data('RGB', image_data.width * 3)
            arr = np.array([int(string_i) for string_i in data])
            arr = np.reshape(arr,(image_data.height, image_data.width, -1))
            #.transpose(1,0,2)
            im = Image.fromarray(arr.astype('uint8'))
            self.gif_buffer.append(im)

            if self.terminal:
                gifname = self.filename.split('.')[0] + '.gif'
                self.viewer.savegif(gifname,arr=self.gif_buffer, duration=self.viz)



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

    def _step(self, action):
        ob, reward, done, info = self.env.step(action)
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
