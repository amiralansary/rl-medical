#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: expreplay.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
# Modified: Amir Alansary <amiralansary@gmail.com>

import six
import copy
import threading
import numpy as np
from six.moves import queue, range
from collections import deque, namedtuple

from tensorpack.utils import logger
from tensorpack.dataflow import DataFlow
from tensorpack.utils.stats import StatCounter
from tensorpack.callbacks.base import Callback
from tensorpack.utils.utils import get_tqdm, get_rng
from tensorpack.utils.concurrency import LoopThread, ShareSessionThread

__all__ = ['ExpReplay']


# def initialize_experience(agents):
    # exp_list=[]
    # for i in agents:
    #     exp_list.append('state_{}'.format(i),'action_{}'.format(i),'reward_{}'.format(i),'isOver_{}'.format(i))
Experience = namedtuple('Experience',['state', 'action', 'reward', 'isOver'])


class ReplayMemory(object):
    def __init__(self, max_size, state_shape, history_len,agents):
        self.max_size = int(max_size)
        self.state_shape = state_shape
        self.history_len = int(history_len)
        self.agents=agents

        self.state=np.zeros((self.agents,self.max_size)+state_shape,dtype='uint8')
        self.action=np.zeros((self.agents,self.max_size),dtype='int32')
        self.reward=np.zeros((self.agents,self.max_size),dtype='float32')
        self.isOver=np.zeros((self.agents,self.max_size),dtype='bool')

        self._curr_pos=0
        self._curr_size = 0
        self._hist = deque(maxlen=history_len - 1)

    def append(self, exp):
        """Append the replay memory with experience sample
        Args:
            exp (Experience): experience contains (state, reward, action, isOver)
        """
        # increase current memory size if it is not full yet
        if self._curr_size < self.max_size:
            self._assign(self._curr_pos, exp)
            self._curr_pos = (self._curr_pos + 1) % self.max_size
            self._curr_size += 1
        else:
            self._assign(self._curr_pos, exp)
            self._curr_pos = (self._curr_pos + 1) % self.max_size
        if np.all(exp.isOver):
            self._hist.clear()
        else:
            self._hist.append(exp)

    def recent_state(self):
        """ return a list of (hist_len-1,) + STATE_SIZE """
        lst = list(self._hist)
        states=[]
        for i in range(0,self.agents):
            states_temp=[np.zeros(self.state_shape, dtype='uint8')] * (self._hist.maxlen - len(lst))
            states_temp.extend([k.state[i] for k in lst])
            states.append(states_temp)

        return states

    def sample(self, idx):
        """ Sample an experience replay from memory with index idx
        :returns: a tuple of (state, reward, action, isOver)
                  where state is of shape STATE_SIZE + (history_length+1,)
        """
        idx = (self._curr_pos + idx) % self._curr_size
        k = self.history_len + 1

        states=[]
        rewards=[]
        actions=[]
        isOver=[]
        for i in range(0,self.agents):
            if idx + k <= self._curr_size:
                states.append(self.state[i,idx: idx + k])
                rewards.append(self.reward[i,idx: idx + k])
                actions.append(self.action[i,idx: idx + k])
                isOver.append(self.isOver[i,idx: idx + k])
            else:
                end = idx + k - self._curr_size
                states.append(self._slice(self.state[i],idx,end))
                rewards.append(self._slice(self.reward[i],idx,end))
                actions.append(self._slice(self.action[i],idx,end))
                isOver.append(self._slice(self.isOver[i],idx,end))

        ret = self._pad_sample(states,rewards,actions,isOver)
        return ret

    # the next_state is a different episode if current_state.isOver==True
    def _pad_sample(self, states,rewards,actions,isOver):
        for k in range(self.history_len - 2, -1, -1):
            for i in range(0,self.agents):
                if isOver[i][k]:
                    states[i]=copy.deepcopy(states[i])
                    states[i][:k + 1].fill(0)
            break

        # transpose state
        rewards_out=[]
        actions_out=[]
        isOver_out=[]
        for i in range(0,self.agents):
            if states[i].ndim==4: # 3D States
                states[i]=states[i].transpose(1,2,3,0)
            else: # 2D States
                states[i]=states[i].transpose(1,2,0)
            rewards_out.append(rewards[i][-2])
            actions_out.append(actions[i][-2])
            isOver_out.append(isOver[i][-2])

        return states, rewards_out, actions_out,isOver_out

    def _slice(self, arr, start, end):
        s1 = arr[start:]
        s2 = arr[:end]
        return np.concatenate((s1, s2), axis=0)

    def __len__(self):
        return self._curr_size

    def _assign(self, pos, exp):
        for i in range (0,self.agents):
            self.state[i,pos]=exp.state[i]
            self.action[i,pos]=exp.action[i]
            self.reward[i,pos]=exp.reward[i]
            self.isOver[i,pos]=exp.isOver[i]


###############################################################################

class ExpReplay(DataFlow, Callback):
    """
    Implement experience replay in the paper
    `Human-level control through deep reinforcement learning
    <http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html>`_.

    This implementation provides the interface as a :class:`DataFlow`.
    This DataFlow is __not__ fork-safe (thus doesn't support multiprocess prefetching).

    This implementation assumes that state is
    batch-able, and the network takes batched inputs.
    """

    def __init__(self,
                 predictor_io_names,
                 player,
                 state_shape,
                 batch_size,
                 memory_size, init_memory_size,
                 init_exploration,
                 update_frequency, history_len,agents=2):
        """
        Args:
            predictor_io_names (tuple of list of str): input/output names to
                predict Q value from state.
            player (RLEnvironment): the player.
            update_frequency (int): number of new transitions to add to memory
                after sampling a batch of transitions for training.
            history_len (int): length of history frames to concat. Zero-filled
                initial frames.
        """
        init_memory_size = int(init_memory_size)
        # for k, v in locals().items():
        for k, v in list(locals().items()):
            if k != 'self':
                setattr(self, k, v)
        self.agents = agents

        self.exploration = init_exploration
        self.num_actions = player.action_space.n
        logger.info("Number of Legal actions: {}".format(self.num_actions))

        self.rng = get_rng(self)
        self._init_memory_flag = threading.Event()  # tell if memory has been initialized

        # a queue to receive notifications to populate memory
        self._populate_job_queue = queue.Queue(maxsize=5)

        self.mem = ReplayMemory(memory_size, state_shape, history_len,agents=self.agents)

        self._current_ob = self.player.reset()

        self._player_scores=[]
        self._player_distError=[]
        for i in range(0,self.agents):
            self._player_scores.append(StatCounter())
            self._player_distError.append(StatCounter())

    def get_simulator_thread(self):
        # spawn a separate thread to run policy
        def populate_job_func():
            self._populate_job_queue.get()
            for _ in range(self.update_frequency):
                self._populate_exp()

        th = ShareSessionThread(LoopThread(populate_job_func, pausable=False))
        th.name = "SimulatorThread"
        return th

    def _init_memory(self):
        logger.info("Populating replay memory with epsilon={} ...".format(self.exploration))

        with get_tqdm(total=self.init_memory_size) as pbar:
            while len(self.mem) < self.init_memory_size:
                self._populate_exp()
                pbar.update()
        self._init_memory_flag.set()

    # quickly fill the memory for debug
    def _fake_init_memory(self):
        from copy import deepcopy
        with get_tqdm(total=self.init_memory_size) as pbar:
            while len(self.mem) < 5:
                self._populate_exp()
                pbar.update()
            while len(self.mem) < self.init_memory_size:
                self.mem.append(deepcopy(self.mem._hist[0]))
                pbar.update()
        self._init_memory_flag.set()

    def _populate_exp(self):
        """ populate a transition by epsilon-greedy"""
        act=np.zeros(self.agents,)
        history=[]
        old_s=self._current_ob
        q_values = [(0,)*self.num_actions] * self.agents
        if self.rng.rand() <= self.exploration or (len(self.mem) <= self.history_len):
        # initialize q_values to zeros
            for i in range(0, self.agents):
                act[i] = self.rng.choice(range(self.num_actions))

        else:
            # build a history state

            for i in range(0, self.agents):
                history.append(self.mem.recent_state()[i])
                history[i].append(old_s[i])
                if np.ndim(history[i]) == 4:  # 3d states
                   history[i] = np.stack(history[i], axis=3)
                else:
                  history[i] = np.stack(history[i], axis=2)
                history[i]=history[i][None,:,:,:,:]

        # assume batched network - this is the bottleneck
            q_values = self.predictor(*history)
        #     q_values=get_DQN_prediction(history)
            for i in range(0,self.agents):
                act[i] = np.argmax(q_values[i])
        isOver=np.zeros((self.agents,), dtype='bool')

        current_state, reward, isOver, info = self.player.step(act, q_values,isOver)
        self._current_ob=current_state
        for i in range(0,self.agents):
            if isOver[i]:
                self._player_scores[i].feed(info['score_{}'.format(i)])
                self._player_distError[i].feed(info['distError_{}'.format(i)])
            if np.all(isOver):
                self.player.reset()

        self.mem.append(Experience(old_s, act, reward,isOver))

    def _debug_sample(self, sample):
        import cv2

        def view_state(comb_state):
            state = comb_state[:, :, :-1]
            next_state = comb_state[:, :, 1:]
            r = np.concatenate([state[:, :, k] for k in range(self.history_len)], axis=1)
            r2 = np.concatenate([next_state[:, :, k] for k in range(self.history_len)], axis=1)
            r = np.concatenate([r, r2], axis=0)
            cv2.imshow("state", r)
            cv2.waitKey()

        print("Act: ", sample[2], " reward:", sample[1], " isOver: ", sample[3])
        if sample[1] or sample[3]:
            view_state(sample[0])

    def get_data(self):
        # wait for memory to be initialized
        self._init_memory_flag.wait()

        while True:
            idx = self.rng.randint(
                self._populate_job_queue.maxsize * self.update_frequency,
                len(self.mem) - self.history_len - 1,
                size=self.batch_size)
            batch_exp = [self.mem.sample(i) for i in idx]

            yield self._process_batch(batch_exp)
            self._populate_job_queue.put(1)

    def _process_batch(self, batch_exp):
        states = np.asarray([e[0] for e in batch_exp], dtype='uint8')
        rewards = np.asarray([e[1] for e in batch_exp], dtype='float32')
        actions = np.asarray([e[2] for e in batch_exp], dtype='int8')
        isOver = np.asarray([e[3] for e in batch_exp], dtype='bool')
        return [states, actions, rewards,isOver]


    def _setup_graph(self):
        self.predictor = self.trainer.get_predictor(*self.predictor_io_names)

    def _before_train(self):
        self._init_memory()
        self._simulator_th = self.get_simulator_thread()
        self._simulator_th.start()

    def _trigger(self):
        # log player statistics in training
        v = self._player_scores
        dist = self._player_distError
        for i in range(0,self.agents):
            try:
                mean, max = v[i].average, v[i].max
                self.trainer.monitors.put_scalar('expreplay/mean_score_{}'.format(i), mean)
                self.trainer.monitors.put_scalar('expreplay/max_score_{}'.format(i), max)
                mean, max = dist[i].average, dist[i].max
                self.trainer.monitors.put_scalar('expreplay/mean_dist_{}'.format(i), mean)
                self.trainer.monitors.put_scalar('expreplay/max_dist_{}'.format(i), max)
            except Exception:
                logger.exception("Cannot log training scores.")
            v[i].reset()
            dist[i].reset()


        # monitor number of played games and successes of reaching the target
        if self.player.num_games.count:
            self.trainer.monitors.put_scalar('n_games',
                                             np.asscalar(self.player.num_games.sum))
        else:
            self.trainer.monitors.put_scalar('n_games', 0)

        for i in range(0,self.agents):

            if self.player.num_success[i].count :
                self.trainer.monitors.put_scalar('n_success_{}'.format(i),
                                                 np.asscalar(self.player.num_success[i].sum))
                self.trainer.monitors.put_scalar('n_success_ratio_{}'.format(i),
                                             self.player.num_success[i].sum / self.player.num_games.sum)
            else:
                self.trainer.monitors.put_scalar('n_success_{}'.format(i), 0)
                self.trainer.monitors.put_scalar('n_success_ratio_{}'.format(i), 0)


        # reset stats
        self.player.reset_stat()

