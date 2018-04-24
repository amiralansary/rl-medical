#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DQN.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
# Modified: Amir Alansary <amiralansary@gmail.com>

#------------------------------------------------------------------------------
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.simplefilter("ignore", category=PendingDeprecationWarning)
#------------------------------------------------------------------------------

import os
import time
import argparse
import numpy as np
import tensorflow as tf
from collections import deque

from detectPlanePlayerCardio import MedicalPlayer, FrameStack
# from detectPlanePlayerBrain import MedicalPlayer, FrameStack


from tensorpack import (PredictConfig, OfflinePredictor, get_model_loader,
                        logger, TrainConfig, ModelSaver, PeriodicTrigger,
                        ScheduledHyperParamSetter, ObjAttrParam,
                        HumanHyperParamSetter, argscope, RunOp, LinearWrap,
                        FullyConnected, LeakyReLU, PReLU, SimpleTrainer,
                        launch_train_with_config)

from tensorpack.input_source import QueueInput
from tensorpack_medical.models.conv3d import Conv3D

from common import Evaluator, eval_model_multithread, play_n_episodes
from DQNModel import Model3D as DQNModel
from expreplay import ExpReplay

###############################################################################

# BATCH SIZE USED IN NATURE PAPER IS 32 - MEDICAL IS UNKNOWN
BATCH_SIZE = 32
# BREAKOUT (84,84) - MEDICAL 2D (60,60) - MEDICAL 3D (26,26,26)
IMAGE_SIZE = (50, 50, 9) # (85, 85, 9)#(27, 27, 27)
FRAME_HISTORY = 4
## FRAME SKIP FOR ATARI GAMES
# ACTION_REPEAT = 4
# the frequency of updating the target network
UPDATE_FREQ = 4
# DISCOUNT FACTOR - NATURE (0.99) - MEDICAL (0.9)
GAMMA = 0.9 #0.99
# REPLAY MEMORY SIZE - NATURE (1e6) - MEDICAL (1e5 view-patches)
MEMORY_SIZE = 1e6
# consume at least 1e6 * 27 * 27 * 27 bytes
INIT_MEMORY_SIZE = MEMORY_SIZE // 20 #5e4
# each epoch is 100k played frames
STEPS_PER_EPOCH = 10000 // UPDATE_FREQ * 10
# Evaluation episode
EVAL_EPISODE = 50
# MEDICAL DETECTION ACTION SPACE (UP,FORWARD,RIGHT,LEFT,BACKWARD,DOWN)
NUM_ACTIONS = None
# dqn method - double or dual (default: double)
METHOD = None

SPACING = (5,5,5)

###############################################################################
###############################################################################
data_dir = 'data_dir'
train_list = 'list_of_train_filenames.txt'
test_list = 'list_of_test_filenames.txt'

logger_dir = os.path.join('train_log', 'expriment_1')

###############################################################################


def get_player(directory=None, files_list= None,
               viz=False, train=False, savegif=False):

    env = MedicalPlayer(directory=directory, files_list=files_list,
                        screen_dims=IMAGE_SIZE, viz=viz, savegif=savegif,
                        train=train, spacing=SPACING, max_num_frames=500)
    if not train:
        # in training, history is taken care of in expreplay buffer
        env = FrameStack(env, FRAME_HISTORY)
    return env

###############################################################################

class Model(DQNModel):
    def __init__(self):
        super(Model, self).__init__(IMAGE_SIZE, FRAME_HISTORY, METHOD, NUM_ACTIONS, GAMMA)

    def _get_DQN_prediction(self, image):
        """ image: [0,255]"""
        image = image / 255.0
        with argscope(Conv3D,
                      activation=PReLU.symbolic_function,
                      use_bias=True):
            #,argscope(LeakyReLU, alpha=0.01):
            l = (LinearWrap(image)
                 .Conv3D('conv0', out_channel=32,
                         kernel_shape=[8,8,3], stride=[2,2,1])
                 # Nature architecture
                 .Conv3D('conv1', out_channel=32,
                         kernel_shape=[8,8,3], stride=[2,2,1])
                 .Conv3D('conv2', out_channel=64,
                         kernel_shape=[4,4,3], stride=[2,2,1])
                 .Conv3D('conv3', out_channel=64,
                         kernel_shape=[3,3,3], stride=[1,1,1])
                 .FullyConnected('fc0', 512)
                 .tf.nn.leaky_relu(alpha=0.01)
                 .FullyConnected('fc1', 256)
                 .tf.nn.leaky_relu(alpha=0.01)
                 .FullyConnected('fc2', 128)
                 .tf.nn.leaky_relu(alpha=0.01)())
        if 'Dueling' not in self.method:
            Q = FullyConnected('fct', l, self.num_actions, nl=tf.identity)
        else:
            # Dueling DQN
            V = FullyConnected('fctV', l, 1, activation=tf.identity)
            As = FullyConnected('fctA', l, self.num_actions,
                                activation=tf.identity)
            Q = tf.add(As, V - tf.reduce_mean(As, 1, keepdims=True))
        return tf.identity(Q, name='Qvalue')

###############################################################################

def get_config():
    expreplay = ExpReplay(
        predictor_io_names=(['state'], ['Qvalue']),
        player=get_player(directory=data_dir, files_list=train_list,
                          train=True),
        state_shape=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        memory_size=MEMORY_SIZE,
        init_memory_size=INIT_MEMORY_SIZE,
        init_exploration=1.0,
        update_frequency=UPDATE_FREQ,
        history_len=FRAME_HISTORY
    )

    return TrainConfig(
        # dataflow=expreplay,
        data=QueueInput(expreplay),
        model=Model(),
        callbacks=[
            ModelSaver(),
            PeriodicTrigger(
                RunOp(DQNModel.update_target_param, verbose=True),
                # update target network every 10k steps
                every_k_steps=10000 // UPDATE_FREQ),
            expreplay,
            ScheduledHyperParamSetter('learning_rate',
                                      [(60, 4e-4), (100, 2e-4)]),
                                        # [(30, 1e-2), (60, 1e-3), (85, 1e-4), (95, 1e-5)]),
            ScheduledHyperParamSetter(
                ObjAttrParam(expreplay, 'exploration'),
                # 1->0.1 in the first million steps
                [(0, 1), (10, 0.1), (320, 0.01)],
                interp='linear'),
            PeriodicTrigger(
                Evaluator(nr_eval=EVAL_EPISODE, input_names=['state'],
                          output_names=['Qvalue'], get_player_fn=get_player,
                          directory=data_dir, files_list=test_list),
                every_k_epochs=2),
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
                        choices=['DQN', 'Double', 'Dueling', 'DuelingDouble'],
                        default='Double')
    parser.add_argument('--savegif', help='save gif image of the game',
                        action='store_true', default=False)
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # ROM_FILE = args.rom
    METHOD = args.algo
    # set num_actions
    init_player = MedicalPlayer(directory=data_dir,
                                files_list=test_list,
                                screen_dims=IMAGE_SIZE,
                                spacing=SPACING)
    NUM_ACTIONS = init_player.action_space.n
    num_validation_files = init_player.files.num_files

    if args.task != 'train':
        assert args.load is not None
        pred = OfflinePredictor(PredictConfig(
            model=Model(),
            session_init=get_model_loader(args.load),
            input_names=['state'],
            output_names=['Qvalue']))
        if args.task == 'play':
            t0 = time.time()
            play_n_episodes(get_player(directory=data_dir,
                                       files_list=test_list, viz=0.01,
                                       savegif=args.savegif),
                            pred, num_validation_files)

            t1 = time.time()
            print(t1-t0)
        elif args.task == 'eval':
            eval_model_multithread(pred, EVAL_EPISODE, get_player)
    else:
        logger.set_logger_dir(logger_dir) # todo: variable log dir
        config = get_config()
        if args.load:
            config.session_init = get_model_loader(args.load)
        launch_train_with_config(config, SimpleTrainer())
