#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: medical.py
# Author: Amir Alansary <amiralansary@gmail.com>

import os
import six
import random
import threading
import numpy as np
from tensorpack import logger
from collections import (Counter, defaultdict)

import cv2
import time

# import pyglet
from gym import spaces
# from gym.envs.classic_control import rendering

from tensorpack.utils.utils import get_rng
from tensorpack.utils.stats import StatCounter
from tensorpack.RL.envbase import RLEnvironment, DiscreteActionSpace



from sampleTrain import trainFiles


__all__ = ['MedicalPlayer']

_ALE_LOCK = threading.Lock()

# ===================================================================
# =================== 3d medical environment ========================
# ===================================================================

class MedicalPlayer(RLEnvironment):
    """Class that provides 3D medical image environment.
    This is just an implementation of the classic "agent-environment loop".
    Each time-step, the agent chooses an action, and the environment returns
    an observation and a reward."""

    def __init__(self, directory=None, viz=False, screen_dims=(27,27,27), nullop_start=30,location_history_length=10):
        """
        :param train_directory: environment or game name
        :param viz: visualization
            set to 0 to disable
            set to +ve number to be the delay between frames to show
            set to a string to be the directory for storing frames
        :param screen_dims: shape of the frame cropped from the image to feed
            it to dqn (d,w,h) - defaults (27,27,27)
        :param nullop_start: start with random number of null ops
        :param location_history_length: consider lost of lives as end of
            episode (useful for training)
        """
        super(MedicalPlayer, self).__init__()

        # needed for the medical environment
        self.info = None
        self.width, self.height, self.depth = screen_dims
        self._loc_history_length = location_history_length

        with _ALE_LOCK:
            self.rng = get_rng(self)
            # visualization setup
            if isinstance(viz, six.string_types):   # check if viz is a string
                assert os.path.isdir(viz), viz
                # todo implement save states to images
                # self.env.env.ale.setString(b'record_screen_dir', viz)
                # rgb_image = self.env.render('rgb_array')
                viz = 0
            if isinstance(viz, int):
                viz = float(viz)
            self.viz = viz
            if self.viz and isinstance(self.viz, float):
                # self.window = pyglet.window.Window(1024,1024)
                                                   #width=self.width,
                                                   # height=self.height)
                self.viewer = None
                # self._render()
                # self.windowname = os.path.basename(env_name)
                # cv2.startWindowThread()
                # cv2.namedWindow(self.windowname)
                # from gym.envs.classic_control import rendering
                # self.viewer = SimpleImageViewer()


        # circular buffer to store history
        self._loc_history = list([(0,0,0)]) * self._loc_history_length

        self.screen_dims = screen_dims
        self.nullop_start = nullop_start

        self.current_episode_score = StatCounter()
        self.actions = self.getMinimalActionSet()

        self.train_files = trainFiles(directory) # TODO rename trainFiles

        self.sampled_files = self.train_files.sample_circular()
        self.restart_episode()

    def restart_episode(self):
        # reset the stat counter
        self.current_episode_score.reset()
        self.num_games.feed(1)

        # with _ALE_LOCK:
        self.reset_game()
        # # random null-ops start by performing random number of dummy actions at the beginning of each episode
        # n = self.rng.randint(self.nullop_start)
        # self.last_raw_screen = self.get_screen()
        # for k in range(n):
        #     if k == n-1:
        #         self.last_raw_screen = self.get_screen()
        #     self.action(0)

    def reset_game(self):
        """
        reset location history buffer
        """
        self.new_random_game()
        self._loc_history = list([(0,0,0)]) * self._loc_history_length
        self.terminal = False
        # return self.get_screen(location)

    def new_random_game(self):
        # print('\n============== new game ===============\n')
        self.terminal = False
        self.viewer = None
        # sample a new image
        self._game_img, self._target_loc, image_filename = next(self.sampled_files)# self.train_files.sample()

        # image volume size
        self._game_dims = self._game_img.dims
        self._loc_dims = np.array((self.screen_dims[0]+1, self.screen_dims[1]+1, self.screen_dims[2]+1, self._game_dims[0]-self.screen_dims[0]-1, self._game_dims[1]-self.screen_dims[1]-1, self._game_dims[2]-self.screen_dims[2]-1))

        x = self.rng.randint(self._loc_dims[0]+1, self._loc_dims[3]-1)
        y = self.rng.randint(self._loc_dims[1]+1, self._loc_dims[4]-1)
        z = self.rng.randint(self._loc_dims[2]+1, self._loc_dims[5]-1)

        self._location = (x,y,z)
        self._start_location = (x,y,z)
        self._screen = self.get_screen()

        self.cur_dist = np.linalg.norm(self._location - self._target_loc)

    def action(self, act):
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
        current_loc = self._location
        self.terminal = False
        go_out = False

        act = self.actions[act]

        # UP Z+
        if (act==0):
            next_location = (current_loc[0],current_loc[1],current_loc[2]+1)
            if (next_location[2]>=self._loc_dims[5]):
                # print(' trying to go out the image Z+ ',)
                next_location = current_loc
                go_out = True

        # FORWARD Y+
        if (act==1):
            next_location = (current_loc[0],current_loc[1]+1,current_loc[2])
            if (next_location[1]>=self._loc_dims[4]):
                # print(' trying to go out the image Y+ ',)
                next_location = current_loc
                go_out = True

        # RIGHT X+
        if (act==2):
            next_location = (current_loc[0]+1,current_loc[1],current_loc[2])
            if (next_location[0]>=self._loc_dims[3]):
                # print(' trying to go out the image X+ ',)
                next_location = current_loc
                go_out = True

        # LEFT X-
        if (act==3):
            next_location = (current_loc[0]-1,current_loc[1],current_loc[2])
            if (next_location[0]<=self._loc_dims[0]):
                # print(' trying to go out the image X- ',)
                next_location = current_loc
                go_out = True

        # BACKWARD Y-
        if (act==4):
            next_location = (current_loc[0],current_loc[1]-1,current_loc[2])
            if (next_location[1]<=self._loc_dims[1]):
                # print(' trying to go out the image Y- ',)
                next_location = current_loc
                go_out = True

        # DOWN Z-
        if (act==5):
            next_location = (current_loc[0],current_loc[1],current_loc[2]-1)
            if (next_location[2]<=self._loc_dims[2]):
                # print(' trying to go out the image Z- ',)
                next_location = current_loc
                go_out = True

        # punish -1 reward if the agent tries to go out
        if go_out:
            # print('trying to go out')
            # logger.info('Trying to go out \n current location = {} - next_location = {} - action {} - reward = {} - terminal = {}'.format(current_loc, next_location, action, self.reward, self.terminal))
            self.reward = -1
            # self.terminal = True # end episode and restart
        else:
            self.reward = self._calc_reward(current_loc, next_location)

        # check if agent oscillates
        # self._add_loc(next_location)
        # if self._oscillate: done  = True
        # update screen, reward ,location, terminal
        self._location = next_location
        self._screen = self.get_screen()
        self.cur_dist = np.linalg.norm(self._location - self._target_loc)

        # if (np.array(self._location)==np.array(self._target_loc)).all():
        if self.cur_dist<1:
            self.terminal = True
            self.num_success.feed(1)
            # logger.info('Target reached!! \n start location = {} - target_location = {} - reward = {} - terminal = {}'.format(self._start_location, self._target_loc, self.reward, self.terminal))
            # return (self.reward, self.terminal)

        # if self._oscillate:
        #     if (self._location==self._target_loc).any():
        #         self.terminal = True
        #         logger.info('Found target at current location = {} - target location = {} - reward = {} - terminal = {}'.format(self._location, self._target_loc, self.reward, self.terminal))
        #     else:
        #         self.terminal = False
        #         logger.info('Stuck at current location = {} - target location = {} - reward = {} - terminal = {}'.format(self._location, self._target_loc, self.reward, self.terminal))

        # self.current_episode_score.feed(self.reward)

        if self.terminal:
            # logger.info('reward {}, terminal {}, screen '.format(self.reward, self.terminal, self.get_screen()))
            self.finish_episode()
            self.restart_episode()

        return (self.reward, self.terminal)

    def current_state(self):
        """
        :returns: a gray-scale (h, w, d) float ###uint8 image
        """
        screen = self.get_screen() # get the current 3d screen
        with _ALE_LOCK:
            if self.viz:
                if isinstance(self.viz, float):
                    self._render()
            # if isinstance(self.viz, float):
            #     from gym.envs.classic_control import rendering
            #     self.viewer = SimpleImageViewer()
            #     # self._render()
            #     img = screen[:,:,int(self.depth/2)]
            #     print(img.shape)
            #     img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            #     print(img.shape)
            #     self.viewer.imshow(img)
            #     # cv2.imshow(self.windowname, ret)
            #     # time.sleep(10000)

        return screen
        # ret = self.get_screen()
        # # max-pooled over the last screen
        # ret = np.maximum(ret, self.last_raw_screen)
        # if self.viz:
        #     if isinstance(self.viz, float):
        #         self.env.render()
        #         # cv2.imshow(self.windowname, ret)
        #         # time.sleep(self.viz)
        # ret = ret[self.height_range[0]:self.height_range[1],self.height_range[2], :].astype('float32')
        # # 0.299,0.587.0.114. same as rgb2y in torch/image
        # # ret = cv2.cvtColor(ret, cv2.COLOR_RGB2GRAY)
        # ret = cv2.resize(ret, self.image_shape)
        # return ret.astype('uint8')  # to save some memory


    def _add_loc(self, location):
        ''' Add new location points to the location history buffer
        '''
        self._loc_history[:-1] = self._loc_history[1:]
        self._loc_history[-1] = location


    def get_screen(self):

        xmin = self._location[0] - int(self.width/2) - 1
        xmax = self._location[0] + int(self.width/2)
        ymin = self._location[1] - int(self.height/2) - 1
        ymax = self._location[1] + int(self.height/2)
        zmin = self._location[2] - int(self.depth/2) - 1
        zmax = self._location[2] + int(self.depth/2)
        screen = self._game_img.data[xmin:xmax, ymin:ymax, zmin:zmax]
        # logger.info('xmin {} xmax {} xmax-xmin {} screen shape {} - image shape {}'.format(xmin,xmax,xmax-xmin,np.shape(screen),np.shape(self._game_img.data)))
        # logger.info('screen {}'.format(screen))
        # logger.info('xmin {} xmax {} ymin {} ymax {} zmin {} zmax {}'.format(xmin,xmax,ymin,ymax,zmin,zmax))
        # logger.info('self._game_img.data {}'.format(np.shape(self._game_img.data)))
        return screen


    def _calc_reward(self, current_loc, next_loc):
        ''' Calculate the new reward based on the euclidean distance to the target location
        '''
        curr_dist = np.linalg.norm(current_loc - self._target_loc)
        next_dist = np.linalg.norm(next_loc - self._target_loc)
        return curr_dist - next_dist

    @property
    def _oscillate(self):
        ''' Return True if the agent is stuck and oscillating
        '''
        counter = Counter(self._loc_history)
        freq = counter.most_common()

        if freq[0][0] == (0,0,0):
            return False
        elif (freq[0][1]>3):
            return True

    def get_action_space(self):
        ''' return array of integers for actions
        ACTION_MEANING = {
            1 : "UP",       # MOVE Z+
            2 : "FORWARD",  # MOVE Y+
            3 : "RIGHT",    # MOVE X+
            4 : "LEFT",     # MOVE X-
            5 : "BACKWARD", # MOVE Y-
            6 : "DOWN",     # MOVE Z-
        }
        '''
        return DiscreteActionSpace(len(self.actions))

    # @property
    def get_num_actions(self):
        action_space = spaces.Discrete(6)
        return action_space.n

    def getMinimalActionSet(self):
        """
        Returns a list of the minimal actions set to be used.
        """
        return list(range(0,self.get_num_actions()))

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

    def finish_episode(self):
        self.current_episode_score.feed(self.cur_dist)
        # if self.current_episode_score.count:
        self.stats['score'].append(self.current_episode_score.sum)



    def _render(self, return_rgb_array=False):

        if self.viewer is None:
            self.viewer = SimpleImageViewer()

        # get dimensions
        current_point = self._location
        # get image and convert it to pyglet
        screen = self.get_screen() # get the current 3d screen
        img = screen[:,:,int(self.depth/2)] # sample the middle z-plane
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB) # congvert to rgb
        self.viewer.imshow(img)

        # return self.viewer.render(return_rgb_array = mode=='rgb_array')

# =============================================================================
try:
    import pyglet
except ImportError as e:
    reraise(suffix="HINT: you can install pyglet directly via 'pip install pyglet'. But if you really just want to install all Gym dependencies and not have to think about it, 'pip install -e .[all]' or 'pip install gym[all]' will do it.")

class SimpleImageViewer(object):
    ''' Simple image viewer class for rendering images using pyglet'''

    def __init__(self, display=None):
        self.window = None
        self.isopen = False
        self.display = display

    def imshow(self, arr):
        if self.window is None:
            height, width, channels = arr.shape
            self.window = pyglet.window.Window(width=100, height=100, display=self.display)
            self.width = width
            self.height = height
            self.isopen = True
        assert arr.shape == (self.height, self.width, 3), "You passed in an image with the wrong number shape"
        image = pyglet.image.ImageData(self.width, self.height, 'RGB', arr.tobytes(), pitch=self.width * -3)
        pyglet.gl.glScalef(4.0, 4.0, 1.0)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        image.blit(0,0)
        self.window.flip()

    def close(self):
        if self.isopen:
            self.window.close()
            self.isopen = False
    def __del__(self):
        self.close()


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
