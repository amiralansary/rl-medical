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

def play_one_episode(env, func, render=False):
    def predict(s):
        """
        Map from observation to action, with 0.001 greedy.
        """
        act = func(s[None, :, :, :])[0][0].argmax()
        # if random.random() < 0.001:
        #     spc = env.action_space
        #     act = spc.sample()
        return act

    ob = env.reset()
    sum_r = 0
    while True:
        act = predict(ob)
        ob, r, isOver, info = env.step(act)
        if render:
            env.render()
        sum_r += r
        if isOver:
            logger.info('info distError {}'.format(info['distError']))
            return sum_r, info['distError']

###############################################################################

def play_n_episodes(player, predfunc, nr, render=False):
    logger.info("Start Playing ... ")
    for k in range(nr):
        # if k != 0:
        #     player.restart_episode()
        score, ditance_error = play_one_episode(player, predfunc, render=render)
        print("{}/{}, score={} - distError {}".format(k, nr, score, ditance_error))

###############################################################################

def eval_with_funcs(predictors, nr_eval, get_player_fn, directory=None):
    """
    Args:
        predictors ([PredictorBase])
    """
    class Worker(StoppableThread, ShareSessionThread):
        def __init__(self, func, queue, distErrorQueue):
            super(Worker, self).__init__()
            self._func = func
            self.q = queue
            self.q_dist = distErrorQueue

        def func(self, *args, **kwargs):
            if self.stopped():
                raise RuntimeError("stopped!")
            return self._func(*args, **kwargs)

        def run(self):
            with self.default_sess():
                player = get_player_fn(directory=directory,train=False)
                while not self.stopped():
                    try:
                        score, ditance_error = play_one_episode(player, self.func)
                        # print("Score, ", score)
                    except RuntimeError:
                        return
                    self.queue_put_stoppable(self.q, score)
                    self.queue_put_stoppable(self.q_dist, ditance_error)

    q = queue.Queue()
    q_dist = queue.Queue()

    threads = [Worker(f, q, q_dist) for f in predictors]

    for k in threads:
        k.start()
        time.sleep(0.1)  # avoid simulator bugs
    stat = StatCounter()
    dist_stat = StatCounter()

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

def eval_model_multithread(pred, nr_eval, get_player_fn):
    """
    Args:
        pred (OfflinePredictor): state -> Qvalue
    """
    NR_PROC = min(multiprocessing.cpu_count() // 2, 8)
    with pred.sess.as_default():
        mean_score, max_score, mean_dist, max_dist = eval_with_funcs([pred] * NR_PROC, nr_eval, get_player_fn)
    logger.info("Average Score: {}; Max Score: {}; Average Distance: {}; Max Distance: {}".format(mean_score, max_score, mean_dist, max_dist))

###############################################################################

class Evaluator(Callback):

    def __init__(self, nr_eval, input_names,
                 output_names, directory, get_player_fn):
        self.directory = directory
        self.eval_episode = nr_eval
        self.input_names = input_names
        self.output_names = output_names
        self.get_player_fn = get_player_fn

    def _setup_graph(self):
        NR_PROC = min(multiprocessing.cpu_count() // 2, 20)
        self.pred_funcs = [self.trainer.get_predictor(
            self.input_names, self.output_names)] * NR_PROC

    def _trigger(self):
        t = time.time()
        mean_score, max_score, mean_dist, max_dist = eval_with_funcs(self.pred_funcs, self.eval_episode, self.get_player_fn, self.directory)
        t = time.time() - t
        if t > 10 * 60:  # eval takes too long
            self.eval_episode = int(self.eval_episode * 0.94)
        self.trainer.monitors.put_scalar('mean_score', mean_score)
        self.trainer.monitors.put_scalar('max_score', max_score)
        self.trainer.monitors.put_scalar('mean_distance', mean_dist)
        self.trainer.monitors.put_scalar('max_distance', max_dist)


###############################################################################
