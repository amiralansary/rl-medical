#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: common.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
# Modified: Amir Alansary <amiralansary@gmail.com>

import random
import time
import threading
import numpy as np
from tqdm import tqdm
import multiprocessing
from six.moves import queue

# from tensorpack import *
# from tensorpack.utils.stats import *
from tensorpack.utils import logger
# from tensorpack.callbacks import Triggerable
from tensorpack.callbacks.base import Callback
from tensorpack.utils.stats import StatCounter
from tensorpack.utils.utils import get_tqdm_kwargs
from tensorpack.utils.concurrency import (StoppableThread, ShareSessionThread)

import traceback


###############################################################################

def play_one_episode(env, func, render=False,agents=2):
    def predict(s,agents):
        """
        Run a full episode, mapping observation to action, WITHOUT 0.001 greedy.
    :returns sum of rewards
        """
        # pick action with best predicted Q-value
        acts=np.zeros((agents,))
        for i in range(0,agents):
            s[i]=s[i][None,:,:,:]
        q_values = func(*s)
        for i in range(0,agents):
            q_values[i] = q_values[i].flatten()
            acts[i] = np.argmax(q_values[i])

        return acts, q_values

    obs = env.reset()
    obs=list(obs)
    sum_r = np.zeros((agents,))
    filenames_list = []
    distError_list = []
    isOver=[False]*agents
    while True:
        acts, q_values = predict(obs,agents)
        obs,r, isOver, info = env.step(acts, q_values,isOver)
        obs=list(obs)
        if render:
            env.render()

        for i in range(0,agents):
            if not isOver[i]:
                sum_r[i] += r[i]
            if np.all(isOver):
                filenames_list.append(info['filename_{}'.format(i)])
                distError_list.append(info['distError_{}'.format(i)])
        if np.all(isOver):
            return sum_r, filenames_list,distError_list, q_values


###############################################################################

def play_n_episodes(player, predfunc, nr, render=False,agents=2):
    """wraps play_one_episode, playing a single episode at a time and logs results
    used when playing demos."""
    logger.info("Start Playing ... ")
    dists=np.zeros((agents,nr))
    for k in range(nr):
        # if k != 0:
        #     player.restart_episode()
        score, filename, distance_error, q_values = play_one_episode(player,
                                                                    predfunc,
                                                                    render=render,agents=agents)
        for i in range(0,agents):
            dists[i,k]=distance_error[i]

            logger.info(
                "{}/{} - {} - AGENT {} - score {} - distError {} - q_values {}".format(k + 1, nr, filename[i],i, score[i], distance_error[i],
                                                                           q_values[i]))
    for i in range(0,agents):
        mean_dists=np.mean(dists[i])
        var_dist=np.var(dists[i])
        logger.info('MEAN DISTANCE OF AGENT {} is {}'.format(i,mean_dists))
        logger.info('VARIANCE DISTANCE OF AGENT {} is {}'.format(i, var_dist))

###############################################################################

def eval_with_funcs(predictors, nr_eval, get_player_fn, files_list=None,agents=2):
    """
    Args:
        predictors ([PredictorBase])

    Runs episodes in parallel, returning statistics about the model performance.
    """

    class Worker(StoppableThread, ShareSessionThread):
        def __init__(self, func, queue, distErrorQueue , agents=2):
            super(Worker, self).__init__()
            self.agents=agents
            self._func = func
            self.q = queue
            self.q_dist = distErrorQueue

        def func(self, *args, **kwargs):
            if self.stopped():
                raise RuntimeError("stopped!")
            return self._func(*args, **kwargs)

        def run(self):
            with self.default_sess():
                player = get_player_fn(task=False,
                                       files_list=files_list,agents=self.agents)
                while not self.stopped():
                    try:
                        sum_r, filename, dist, q_values = play_one_episode(player, self.func,agents=self.agents)
                        # print("Score, ", score)
                    except RuntimeError:
                        return
                    for i in range (0,self.agents):
                        self.queue_put_stoppable(self.q, sum_r[i])
                        self.queue_put_stoppable(self.q_dist, dist[i])


    q = queue.Queue()
    q_dist = queue.Queue()

    threads = [Worker(f, q, q_dist,agents=agents) for f in predictors]

    # start all workers
    for k in threads:
        k.start()
        time.sleep(0.1)  # avoid simulator bugs
    stat = StatCounter()
    dist_stat = StatCounter()

    # show progress bar w/ tqdm
    for _ in tqdm(range(nr_eval), **get_tqdm_kwargs()):
        r = q.get()
        stat.feed(r)
        dist = q_dist.get()
        dist_stat.feed(dist)

    logger.info("Waiting for all the workers to finish the last run...")
    for k in threads:
        k.stop()
    for k in threads:
        k.join()
    while q.qsize():
        r = q.get()
        stat.feed(r)

    while q_dist.qsize():
        dist = q_dist.get()
        dist_stat.feed(dist)

    if stat.count > 0:
        return (stat.average, stat.max, dist_stat.average, dist_stat.max)
    return (0, 0, 0, 0)


###############################################################################

def eval_model_multithread(pred, nr_eval, get_player_fn, files_list):
    """
    Args:
        pred (OfflinePredictor): state -> Qvalue

    Evaluate pretrained models, or checkpoints of models during training
    """
    NR_PROC = min(multiprocessing.cpu_count() // 2, 8)
    with pred.sess.as_default():
        mean_score, max_score, mean_dist, max_dist = eval_with_funcs(
            [pred] * NR_PROC, nr_eval, get_player_fn, files_list)
    logger.info("Average Score: {}; Max Score: {}; Average Distance: {}; Max Distance: {}".format(mean_score, max_score, mean_dist, max_dist))

###############################################################################

class Evaluator(Callback):

    def __init__(self, nr_eval, input_names, output_names,
                 get_player_fn, files_list=None,agents=2):
        self.files_list = files_list
        self.eval_episode = nr_eval
        self.input_names = input_names
        self.output_names = output_names
        self.get_player_fn = get_player_fn
        self.agents=agents

    def _setup_graph(self):
        NR_PROC = min(multiprocessing.cpu_count() // 2, 20)
        self.pred_funcs = [self.trainer.get_predictor(
            self.input_names, self.output_names)] * NR_PROC

    def _trigger(self):
        """triggered by Trainer"""
        t = time.time()
        mean_score, max_score, mean_dist, max_dist = eval_with_funcs(
            self.pred_funcs, self.eval_episode, self.get_player_fn, self.files_list,agents=self.agents)
        t = time.time() - t
        if t > 10 * 60:  # eval takes too long
            self.eval_episode = int(self.eval_episode * 0.94)

        # log scores
        self.trainer.monitors.put_scalar('mean_score', mean_score)
        self.trainer.monitors.put_scalar('max_score', max_score)
        self.trainer.monitors.put_scalar('mean_distance', mean_dist)
        self.trainer.monitors.put_scalar('max_distance', max_dist)

###############################################################################
