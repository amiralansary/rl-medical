#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DQNModel.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
# Modified: Amir Alansary <amiralansary@gmail.com>

import abc
import tensorflow as tf
from tensorpack import ModelDesc, InputDesc
from tensorpack.utils import logger
from tensorpack.tfutils import (
    collection, summary, get_current_tower_context, optimizer, gradproc)
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope



class Model2D(ModelDesc):
    def __init__(self, image_shape, channel, method, num_actions, gamma):
        """
        :param image_shape: the shape of input 2d image
        :param channel: history length and goes to channel dimension in kernel
        :param method: dqn or double (default is double)
        :param num_actions: number of actions
        :param gamma: discount factor
        """
        self.gamma = gamma
        self.method = method
        self.channel = channel
        self.image_shape = image_shape
        self.num_actions = num_actions

    def inputs(self):
        # Use a combined state for efficiency.
        # The first h channels are the current state, and the last h channels are the next state.
        return [InputDesc(tf.uint8,
                          (None,) + self.image_shape + (self.channel + 1,),
                          'comb_state'),
                InputDesc(tf.int64, (None,), 'action'),
                InputDesc(tf.float32, (None,), 'reward'),
                InputDesc(tf.bool, (None,), 'isOver')]

    @abc.abstractmethod
    def _get_DQN_prediction(self, image):
        pass

    # decorate the function
    @auto_reuse_variable_scope
    def get_DQN_prediction(self, image):
        return self._get_DQN_prediction(image)

    def build_graph(self, *inputs):
        comb_state, action, reward, isOver = inputs
        comb_state = tf.cast(comb_state, tf.float32)
        state = tf.slice(comb_state, [0, 0, 0, 0], [-1, -1, -1, self.channel], name='state')
        self.predict_value = self.get_DQN_prediction(state)
        if not get_current_tower_context().is_training:
            return

        reward = tf.clip_by_value(reward, -1, 1)
        next_state = tf.slice(comb_state, [0, 0, 0, 1], [-1, -1, -1, self.channel], name='next_state')
        action_onehot = tf.one_hot(action, self.num_actions, 1.0, 0.0)

        pred_action_value = tf.reduce_sum(self.predict_value * action_onehot, 1)  # N,
        max_pred_reward = tf.reduce_mean(tf.reduce_max(
            self.predict_value, 1), name='predict_reward')
        summary.add_moving_summary(max_pred_reward)

        with tf.variable_scope('target'):
            targetQ_predict_value = self.get_DQN_prediction(next_state)    # NxA

        if 'Double' not in self.method:
            # DQN or Dueling
            best_v = tf.reduce_max(targetQ_predict_value, 1)    # N,
        else:
            # Double-DQN or DuelingDouble
            next_predict_value = self.get_DQN_prediction(next_state)
            self.greedy_choice = tf.argmax(next_predict_value, 1)   # N,
            predict_onehot = tf.one_hot(self.greedy_choice, self.num_actions, 1.0, 0.0)
            best_v = tf.reduce_sum(targetQ_predict_value * predict_onehot, 1)

        target = reward + (1.0 - tf.cast(isOver, tf.float32)) * self.gamma * tf.stop_gradient(best_v)

        cost = tf.losses.huber_loss(target, pred_action_value,
                                    reduction=tf.losses.Reduction.MEAN)
        summary.add_param_summary(('conv.*/W', ['histogram', 'rms']),
                                  ('fc.*/W', ['histogram', 'rms']))   # monitor all W
        summary.add_moving_summary(cost)
        return cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate',initializer=1e-3, trainable=False)
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)
        return optimizer.apply_grad_processors(
            opt, [gradproc.GlobalNormClip(10), gradproc.SummaryGradient()])

    @staticmethod
    def update_target_param():
        vars = tf.global_variables()
        ops = []
        G = tf.get_default_graph()
        for v in vars:
            target_name = v.op.name
            if target_name.startswith('target'):
                new_name = target_name.replace('target/', '')
                logger.info("{} <- {}".format(target_name, new_name))
                ops.append(v.assign(G.get_tensor_by_name(new_name + ':0')))
        return tf.group(*ops, name='update_target_network')



class Model3D(ModelDesc):
    def __init__(self, image_shape, channel, method, num_actions, gamma):
        """
        :param image_shape: the shape of input 3d image
        :param channel: history length and goes to channel dimension in kernel
        :param method: dqn or double (default is double)
        :param num_actions: number of actions
        :param gamma: discount factor

        See http://tensorpack.readthedocs.io/tutorial/training-interface.html for Mode lDesc documentation.
        """
        self.gamma = gamma
        self.method = method
        self.channel = channel
        self.image_shape = image_shape
        self.num_actions = num_actions

    def inputs(self):
        # Use a combined state for efficiency.
        # The first h channels are the current state, and the last h channels are the next state.
        return [InputDesc(tf.uint8,
                          (None,) + self.image_shape + (self.channel + 1,),
                          'comb_state'),
                InputDesc(tf.int64, (None,), 'action'),
                InputDesc(tf.float32, (None,), 'reward'),
                InputDesc(tf.bool, (None,), 'isOver')]

    @abc.abstractmethod
    def _get_DQN_prediction(self, image):
        """this method is overridden in DQN.py, where it will return a list of predicted Q-values"""
        pass

    # decorate the function
    @auto_reuse_variable_scope
    def get_DQN_prediction(self, image):
        return self._get_DQN_prediction(image)

    def build_graph(self, *inputs):
        comb_state, action, reward, isOver = inputs
        comb_state = tf.cast(comb_state, tf.float32)
        state = tf.slice(comb_state, [0, 0, 0, 0, 0], [-1, -1, -1, -1, self.channel], name='state')
        self.predict_value = self.get_DQN_prediction(state)
        if not get_current_tower_context().is_training:
            return

        reward = tf.clip_by_value(reward, -1, 1)
        next_state = tf.slice(comb_state, [0, 0, 0, 0, 1], [-1, -1, -1, -1, self.channel], name='next_state')
        action_onehot = tf.one_hot(action, self.num_actions, 1.0, 0.0)

        pred_action_value = tf.reduce_sum(self.predict_value * action_onehot, 1)  # N,
        max_pred_reward = tf.reduce_mean(tf.reduce_max(
            self.predict_value, 1), name='predict_reward')
        summary.add_moving_summary(max_pred_reward)

        with tf.variable_scope('target'):
            targetQ_predict_value = self.get_DQN_prediction(next_state)  # NxA

        if 'Double' not in self.method:
            # DQN or Dueling
            best_v = tf.reduce_max(targetQ_predict_value, 1)  # N,
        else:
            # Double-DQN or DuelingDouble
            next_predict_value = self.get_DQN_prediction(next_state)
            self.greedy_choice = tf.argmax(next_predict_value, 1)  # N,
            predict_onehot = tf.one_hot(self.greedy_choice, self.num_actions, 1.0, 0.0)
            best_v = tf.reduce_sum(targetQ_predict_value * predict_onehot, 1)

        target = reward + (1.0 - tf.cast(isOver, tf.float32)) * self.gamma * tf.stop_gradient(best_v)
        cost = tf.losses.huber_loss(target, pred_action_value,
                                    reduction=tf.losses.Reduction.MEAN)
        summary.add_param_summary(('conv.*/W', ['histogram', 'rms']),
                                  ('fc.*/W', ['histogram', 'rms']))  # monitor all W
        summary.add_moving_summary(cost)
        return cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=1e-3, trainable=False)
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)
        return optimizer.apply_grad_processors(
            opt, [gradproc.GlobalNormClip(10), gradproc.SummaryGradient()])

    @staticmethod
    def update_target_param():
        """periodically triggered by trainer"""
        vars = tf.global_variables()
        ops = []
        G = tf.get_default_graph()
        for v in vars:
            target_name = v.op.name
            if target_name.startswith('target'):
                new_name = target_name.replace('target/', '')
                logger.info("{} <- {}".format(target_name, new_name))
                ops.append(v.assign(G.get_tensor_by_name(new_name + ':0')))
        return tf.group(*ops, name='update_target_network')
