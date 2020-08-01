#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DQN.py
# Author: Amir Alansary <amiralansary@gmail.com>

import warnings
from evaluator import Evaluator
from logger import Logger
from trainer import Trainer
from DQNModel import DQN
from medical import MedicalPlayer, FrameStack
import argparse
import os
import torch


def warn(*args, **kwargs):
    pass


warnings.warn = warn
warnings.simplefilter("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


###############################################################################
# BREAKOUT (84,84) - MEDICAL 2D (60,60) - MEDICAL 3D (26,26,26)
IMAGE_SIZE = (45, 45, 45)
# how many frames to keep
# in other words, how many observations the network can see
FRAME_HISTORY = 4
# DISCOUNT FACTOR - NATURE (0.99) - MEDICAL (0.9)
GAMMA = 0.9  # 0.99
# num training epochs in between model evaluations
EPOCHS_PER_EVAL = 2
# the number of episodes to run during evaluation
EVAL_EPISODE = 50

###############################################################################


def get_player(directory=None, files_list=None, landmark_ids=None, viz=False,
               task="play", file_type="brain", saveGif=False, saveVideo=False,
               multiscale=True, history_length=20, agents=1, logger=None):
    env = MedicalPlayer(
        directory=directory,
        screen_dims=IMAGE_SIZE,
        viz=viz,
        saveGif=saveGif,
        saveVideo=saveVideo,
        task=task,
        files_list=files_list,
        file_type=file_type,
        landmark_ids=landmark_ids,
        history_length=history_length,
        multiscale=multiscale,
        agents=agents,
        logger=logger)
    if task != "train":
        # in training, env will be decorated by ExpReplay, and history
        # is taken care of in expreplay buffer
        # otherwise, FrameStack modifies self.step to save observations into a
        # queue
        env = FrameStack(env, FRAME_HISTORY, agents)
    return env

###############################################################################
###############################################################################


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--load', help='Path to the model to load')
    parser.add_argument(
        '--task',
        help='''task to perform,
                must load a pretrained model if task is "play" or "eval"''',
        choices=['play', 'eval', 'train'], default='train')
    parser.add_argument(
        '--file_type', help='Type of the training and validation files',
        choices=['brain', 'cardiac', 'fetal'], default='train')
    parser.add_argument(
        '--files', type=argparse.FileType('r'), nargs='+',
        help="""Filepath to the text file that contains list of images.
                Each line of this file is a full path to an image scan.
                For (task == train or eval) there should be two input files
                ['images', 'landmarks']""")
    parser.add_argument(
        '--val_files', type=argparse.FileType('r'), nargs='+',
        help="""Filepath to the text file that contains list of validation
                images. Each line of this file is a full path to an image scan.
                For (task == train or eval) there should be two input files
                ['images', 'landmarks']""")
    parser.add_argument('--saveGif', help='Save gif image of the game',
                        action='store_true', default=False)
    parser.add_argument('--saveVideo', help='Save video of the game',
                        action='store_true', default=False)
    parser.add_argument(
        '--logDir', help='Store logs in this directory during training',
        default='runs', type=str)
    parser.add_argument(
        '--landmarks', nargs='*', help='Landmarks to use in the images',
        type=int, default=[1])
    parser.add_argument(
        '--model_name', help='Models implemented are: Network3d, CommNet',
        default="CommNet", choices=['CommNet', 'Network3d'], type=str)
    parser.add_argument(
        '--batch_size', help='Size of each batch', default=64, type=int)
    parser.add_argument(
        '--memory_size',
        help="""Number of transitions stored in exp replay buffer.
                If too much is allocated training may abruptly stop.""",
        default=1e5, type=int)
    parser.add_argument(
        '--init_memory_size',
        help='Number of transitions stored in exp replay before training',
        default=3e4, type=int)
    parser.add_argument(
        '--max_episodes', help='"Number of episodes to train for"',
        default=1e5, type=int)
    parser.add_argument(
        '--steps_per_episode', help='Maximum steps per episode',
        default=200, type=int)
    parser.add_argument(
        '--target_update_freq',
        help='Number of epochs between each target network update',
        default=10, type=int)
    parser.add_argument(
        '--save_freq', help='Saves network every save_freq steps',
        default=1000, type=int)
    parser.add_argument(
        '--delta',
        help="""Amount to decreases epsilon each episode,
                for the epsilon-greedy policy""",
        default=1e-4, type=float)
    parser.add_argument(
        '--viz', help='Size of the window, None for no visualisation',
        default=0.01, type=float)
    parser.add_argument(
        '--multiscale',
        help='Reduces size of voxel around the agent when it oscillates',
        dest='multiscale', action='store_true')
    parser.set_defaults(multiscale=False)
    parser.add_argument(
        '--write', help='Saves the training logs', dest='write',
        action='store_true')
    parser.set_defaults(write=False)
    parser.add_argument(
        '--train_freq',
        help="""Number of agent steps between each training step on one
                mini-batch""",
        default=1, type=int)

    args = parser.parse_args()

    agents = len(args.landmarks)

    # check valid number of agents:
    assert agents > 0

    # check input files
    if args.task == 'play':
        error_message = f"""Wrong input files {len(args.files)} for {args.task}
                            task - should be 1 \'images.txt\' """
        assert len(args.files) == 1, (error_message)
    else:
        error_message = f"""Wrong input files {len(args.files)} for
                            {args.task} task - should be 2 [\'images.txt\',
                            \'landmarks.txt\'] """
        assert len(args.files) == 2, (error_message)

    logger = Logger(args.logDir, args.write, args.save_freq)

    # load files into env to set num_actions, num_validation_files
    # TODO: is this necessary?
    init_player = MedicalPlayer(files_list=args.files,
                                file_type=args.file_type,
                                landmark_ids=args.landmarks,
                                screen_dims=IMAGE_SIZE,
                                # TODO: why is this always play?
                                task='play',
                                agents=agents,
                                logger=logger)
    NUM_ACTIONS = init_player.action_space.n

    if args.task != 'train':
        # TODO: refactor DQN to not have to create both a q_network and
        # target_network
        dqn = DQN(agents, frame_history=FRAME_HISTORY, logger=logger,
                  type=args.model_name)
        model = dqn.q_network
        model.load_state_dict(torch.load(args.load))
        environment = get_player(files_list=args.files,
                                 file_type=args.file_type,
                                 landmark_ids=args.landmarks,
                                 saveGif=args.saveGif,
                                 saveVideo=args.saveVideo,
                                 task=args.task,
                                 agents=agents,
                                 viz=args.viz,
                                 logger=logger)
        evaluator = Evaluator(environment, model, logger, agents,
                              args.steps_per_episode)
        evaluator.play_n_episodes()
    else:  # train model
        environment = get_player(task='train',
                                 files_list=args.files,
                                 file_type=args.file_type,
                                 landmark_ids=args.landmarks,
                                 agents=agents,
                                 viz=args.viz,
                                 multiscale=args.multiscale,
                                 logger=logger)
        eval_env = None
        if args.val_files is not None:
            eval_env = get_player(task='eval',
                                  files_list=args.val_files,
                                  file_type=args.file_type,
                                  landmark_ids=args.landmarks,
                                  agents=agents,
                                  logger=logger)
        trainer = Trainer(environment,
                          eval_env=eval_env,
                          batch_size=args.batch_size,
                          image_size=IMAGE_SIZE,
                          frame_history=FRAME_HISTORY,
                          update_frequency=args.target_update_freq,
                          replay_buffer_size=args.memory_size,
                          init_memory_size=args.init_memory_size,
                          gamma=GAMMA,
                          steps_per_episode=args.steps_per_episode,
                          max_episodes=args.max_episodes,
                          delta=args.delta,
                          logger=logger,
                          model_name=args.model_name,
                          train_freq=args.train_freq,
                          ).train()
