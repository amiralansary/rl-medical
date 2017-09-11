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
import math
import time
from PIL import Image

from gym import spaces
# from gym.envs.classic_control import rendering

from tensorpack.utils.utils import get_rng
from tensorpack.utils.stats import StatCounter
from tensorpack.RL.envbase import RLEnvironment, DiscreteActionSpace



from sampleTrain import trainFiles, trainFiles_cardio
from viewer import SimpleImageViewer


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

    def __init__(self, directory=None, viz=False, screen_dims=(27,27,27), nullop_start=30,location_history_length=10, save_gif=False):
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
        self.save_gif = save_gif

        with _ALE_LOCK:
            self.rng = get_rng(self)
            # visualization setup
            if isinstance(viz, six.string_types):   # check if viz is a string
                assert os.path.isdir(viz), viz
                # todo implement save states to images -> save gif
                # self.env.env.ale.setString(b'record_screen_dir', viz)
                # rgb_image = self.env.render('rgb_array')
                viz = 0
            if isinstance(viz, int):
                viz = float(viz)
            self.viz = viz
            if self.viz and isinstance(self.viz, float):
                self.viewer = None
                self.gif_buffer = []

        # circular buffer to store history
        self._loc_history = list([(0,0,0)]) * self._loc_history_length

        self.screen_dims = screen_dims
        self.nullop_start = nullop_start

        self.current_episode_score = StatCounter()
        self.actions = self.getMinimalActionSet()

        self.train_files = trainFiles_cardio(directory)
        # self.train_files = trainFiles(directory)
        self.filepath = None

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
        self._game_img, self._target_loc, self.filepath = next(self.sampled_files)# self.train_files.sample()
        self.filename = os.path.basename(self.filepath)

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

        if (np.array(self._location)==np.array(self._target_loc)).all():
        # if self.cur_dist<1:
            self.terminal = True
            self.num_success.feed(1)

        # if self._oscillate:
        #     if (self._location==self._target_loc).any():
        #         self.terminal = True
        #         logger.info('Found target at current location = {} - target location = {} - reward = {} - terminal = {}'.format(self._location, self._target_loc, self.reward, self.terminal))
        #     else:
        #         self.terminal = False
        #         logger.info('Stuck at current location = {} - target location = {} - reward = {} - terminal = {}'.format(self._location, self._target_loc, self.reward, self.terminal))

        if self.terminal:
            # logger.info('reward {}, terminal {}, screen '.format(self.reward, self.terminal, self.get_screen()))
            with _ALE_LOCK:
                if self.viz:
                    if isinstance(self.viz, float):
                        self._render()
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
        return screen


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
        return screen

    def get_plane(self,z=0):
        return self._game_img.data[:, :, z]


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
        if self.save_gif:
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
