#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DQN.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
# Modified: Amir Alansary <amiralansary@gmail.com>

import numpy as np

import os
import sys
# import re
import time
import random
import argparse
import subprocess
import multiprocessing
import threading
from collections import deque

import tensorflow as tf

from medical import MedicalPlayer

from tensorpack import (PredictConfig, get_model_loader, logger, TrainConfig,
                        ModelSaver, PeriodicTrigger, ScheduledHyperParamSetter,
                        ObjAttrParam, HumanHyperParamSetter, QueueInputTrainer,
                        argscope, RunOp, LinearWrap, FullyConnected, LeakyReLU,
                        PReLU)

from tensorpack_medical.RL.common import (MapPlayerState, PreventStuckPlayer,
                           LimitLengthPlayer)

from common import play_model, Evaluator, eval_model_multithread
from DQNModel import Model3D as DQNModel
from expreplay import ExpReplay

from tensorpack_medical.RL.history import HistoryFramePlayer
from tensorpack_medical.models.conv3d import Conv3D

###############################################################################

# BATCH SIZE USED IN NATURE PAPER IS 32 - MEDICAL IS UNKNOWN
BATCH_SIZE = 32 #64
# BREAKOUT (84,84) - MEDICAL 2D (60,60) - MEDICAL 3D (26,26,26)
IMAGE_SIZE = (27, 27, 27)
FRAME_HISTORY = 4
## FRAME SKIP FOR ATARI GAMES
ACTION_REPEAT = 4
# the frequency of updating the target network
UPDATE_FREQ = 4
# DISCOUNT FACTOR - NATURE (0.99) - MEDICAL (0.9)
GAMMA = 0.9 #0.99
# REPLAY MEMORY SIZE - NATURE (1e6) - MEDICAL (1e5 view-patches)
MEMORY_SIZE = 1e6
# consume at least 1e6 * 27 * 27 * 27 bytes
INIT_MEMORY_SIZE = 5e4
# each epoch is 100k played frames
STEPS_PER_EPOCH = 10000 // UPDATE_FREQ * 10
# Evaluation episode
EVAL_EPISODE = 50
# MEDICAL DETECTION ACTION SPACE (UP,FORWARD,RIGHT,LEFT,BACKWARD,DOWN)
NUM_ACTIONS = None
# dqn method - double or dual (default: double)
METHOD = None

TRAIN_DIR = '/vol/medic01/users/aa16914/projects/DQN-landmark/train_files_svr/'
VALID_DIR = '/vol/medic01/users/aa16914/projects/DQN-landmark/validate_files_svr'

###############################################################################

def get_player(directory=None, viz=False, train=False):
    pl = MedicalPlayer(directory=directory,screen_dims=IMAGE_SIZE)

    if not train:
        # create a new axis to stack history on
        # 2d states
        # agent = MapPlayerState(agent, lambda im: im[:, :, np.newaxis])
        pl = MapPlayerState(pl, lambda im: im[:, :, :, np.newaxis]) # 3d states
        # in training, history is taken care of in experience replay buffer
        pl = HistoryFramePlayer(pl, hist_len=FRAME_HISTORY, concat_axis=3)
        pl = PreventStuckPlayer(pl, 30, 1)

    pl = LimitLengthPlayer(pl, 300) # atari LimitLengthAgent(agent, 30000)

    return pl

###############################################################################

class Model(DQNModel):
    def __init__(self):
        super(Model, self).__init__(IMAGE_SIZE, FRAME_HISTORY, METHOD, NUM_ACTIONS, GAMMA)

    def _get_DQN_prediction(self, image):
        """ image: [0,255]"""
        image = image / 255.0
        with argscope(Conv3D, nl=PReLU.symbolic_function, use_bias=True), \
                argscope(LeakyReLU, alpha=0.01):
            l = (LinearWrap(image)
                 # Nature architecture
                 .Conv3D('conv0', out_channel=32, kernel_shape=8, stride=4)
                 .Conv3D('conv1', out_channel=64, kernel_shape=4, stride=2)
                 .Conv3D('conv2', out_channel=64, kernel_shape=3)
                 .FullyConnected('fc0', 512, nl=LeakyReLU)())
        if self.method != 'Dueling':
            Q = FullyConnected('fct', l, self.num_actions, nl=tf.identity)
        else:
            # Dueling DQN
            V = FullyConnected('fctV', l, 1, nl=tf.identity)
            As = FullyConnected('fctA', l, self.num_actions, nl=tf.identity)
            Q = tf.add(As, V - tf.reduce_mean(As, 1, keep_dims=True))
        return tf.identity(Q, name='Qvalue')

###############################################################################

def get_config():
    expreplay = ExpReplay(
        predictor_io_names=(['state'], ['Qvalue']),
        player=get_player(directory=TRAIN_DIR, train=True),
        state_shape=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        memory_size=MEMORY_SIZE,
        init_memory_size=INIT_MEMORY_SIZE,
        init_exploration=1.0,
        update_frequency=UPDATE_FREQ,
        history_len=FRAME_HISTORY
    )

    return TrainConfig(
        dataflow=expreplay,
        model=Model(),
        callbacks=[
            ModelSaver(),
            PeriodicTrigger(
                RunOp(DQNModel.update_target_param, verbose=True),
                every_k_steps=10000 // UPDATE_FREQ),    # update target network every 10k steps
            expreplay,
            ScheduledHyperParamSetter('learning_rate',
                                      [(60, 4e-4), (100, 2e-4)]),
            ScheduledHyperParamSetter(
                ObjAttrParam(expreplay, 'exploration'),
                [(0, 1), (10, 0.1), (320, 0.01)],   # 1->0.1 in the first million steps
                interp='linear'),
            PeriodicTrigger(
                Evaluator(nr_eval=EVAL_EPISODE, input_names=['state'],
                          output_names=['Qvalue'], directory=VALID_DIR,
                          get_player_fn=get_player),
                every_k_epochs=10),
            HumanHyperParamSetter('learning_rate'),
        ],
        steps_per_epoch=STEPS_PER_EPOCH,
        max_epoch=1000,
    )



###############################################################################
###############################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--task', help='task to perform',
                        choices=['play', 'eval', 'train'], default='train')
    # parser.add_argument('--rom', help='atari rom', required=True)
    parser.add_argument('--algo', help='algorithm',
                        choices=['DQN', 'Double', 'Dueling'], default='Double')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # ROM_FILE = args.rom
    METHOD = args.algo
    # set num_actions
    NUM_ACTIONS = MedicalPlayer(directory=TRAIN_DIR,screen_dims=IMAGE_SIZE).get_action_space().num_actions()
    # NUM_ACTIONS = MedicalPlayer(ROM_FILE).get_action_space().num_actions()
    # logger.info("ROM: {}, Num Actions: {}".format(ROM_FILE, NUM_ACTIONS))

    if args.task != 'train':
        assert args.load is not None
        cfg = PredictConfig(
            model=Model(),
            session_init=get_model_loader(args.load),
            input_names=['state'],
            output_names=['Qvalue'])
        if args.task == 'play':
            play_model(cfg, get_player(viz=0.01))
        elif args.task == 'eval':
            eval_model_multithread(cfg, EVAL_EPISODE, get_player)
    else:
        # todo: variable log dir
        logger.set_logger_dir( 'train_log/DQN'
            # os.path.join('train_log', 'DQN-{}'.format(
            #     os.path.basename(ROM_FILE).split('.')[0]))
            )

        config = get_config()
        if args.load:
            config.session_init = get_model_loader(args.load)
        QueueInputTrainer(config).train()
