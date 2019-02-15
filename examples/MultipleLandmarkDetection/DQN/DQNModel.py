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
from tensorpack.models.regularize import regularize_cost_from_collection



class Model3D(ModelDesc):
    def __init__(self, image_shape, channel, method, num_actions, gamma,agents=2):
        """
        :param image_shape: the shape of input 3d image
        :param channel: history length and goes to channel dimension in kernel
        :param method: dqn or double (default is double)
        :param num_actions: number of actions
        :param gamma: discount factor

        See http://tensorpack.readthedocs.io/tutorial/training-interface.html for Mode lDesc documentation.
        """
        self.agents=agents
        self.gamma = gamma
        self.method = method
        self.channel = channel
        self.image_shape = image_shape
        self.num_actions = num_actions


    def _get_inputs(self):
        # Use a combined state for efficiency.
        # The first h channels are the current state, and the last h channels are the next state.
        return [InputDesc(tf.uint8,
                          (None,) + (self.agents,)+self.image_shape + (self.channel + 1,),
                          'comb_state'),
                InputDesc(tf.int64, (None,self.agents), 'action'),
                InputDesc(tf.float32, (None,self.agents), 'reward'),
                InputDesc(tf.bool, (None,self.agents), 'isOver'),
                ]

    @abc.abstractmethod
    def _get_DQN_prediction(self, image):
        """this method is overridden in DQN.py, where it will return a list of predicted Q-values"""
        pass

    # decorate the function
    @auto_reuse_variable_scope
    def get_DQN_prediction(self, images):
        return self._get_DQN_prediction(images)

    def _build_graph(self, inputs):
        comb_state, action ,reward, isOver= inputs
        comb_state = tf.cast(comb_state, tf.float32)  # agent 1
        states=[]
        for i in range(0,self.agents):
            states.append( tf.slice(comb_state[:,i,:,:,:,:], [0, 0, 0, 0, 0], [-1, -1, -1, -1, self.channel], name='state_{}'.format(i)))  # agent 1

        self.predict_values = self.get_DQN_prediction(states)
        if not get_current_tower_context().is_training:
            return

        rewards=[]
        next_states=[]
        action_onehot=[]
        pred_action_value=[]
        max_pred_rewards=[]
        for i in range(0,self.agents):
            rewards.append(tf.clip_by_value(reward[:,i], -1, 1))
            next_states.append(tf.slice(comb_state[:,i,:,:,:,:], [0, 0, 0, 0, 1], [-1, -1, -1, -1, self.channel], name='next_state_{}'.format(i)))
            action_onehot.append(tf.one_hot(action[:,i], self.num_actions, 1.0, 0.0))
            pred_action_value.append(tf.reduce_sum(self.predict_values[i] * action_onehot[i], 1)) #N
            max_pred_rewards.append(tf.reduce_mean(tf.reduce_max(self.predict_values[i], 1), name='predict_reward_{}'.format(i)))
            summary.add_moving_summary(max_pred_rewards[i])


        with tf.variable_scope('targets'):
            targetQ_predict_values = self.get_DQN_prediction(next_states)  # NxA   agent 1, agent 2

        best_v = []
        if 'Double' not in self.method:
            # DQN or Dueling

            for i in range(0,self.agents):
                best_v.append(tf.reduce_max(targetQ_predict_values[i], 1))  # N, agent 1

        else:
            # Double-DQN or DuelingDouble
            # raise (' not implemented for multi agent ')
            # self.greedy_choice=[]
            next_predict_values = self.get_DQN_prediction(next_states)
            # predict_onehot=[]
            for i in range(0,self.agents):
                self.greedy_choice = tf.argmax(next_predict_values[i], 1) # N,
                predict_onehot = tf.one_hot(self.greedy_choice, self.num_actions, 1.0, 0.0)
                best_v.append(tf.reduce_sum(targetQ_predict_values[i] * predict_onehot, 1))
        targets=[]
        self.costs=[]
        self.cost=0.0
        for i in range(0,self.agents):
            targets.append(rewards[i] + (1.0 - tf.cast(isOver[:,i], tf.float32)) * self.gamma * tf.stop_gradient(best_v[i]))  # agent 1
            self.costs.append(tf.losses.huber_loss(targets[i], pred_action_value[i],
                                         reduction=tf.losses.Reduction.MEAN))  # agent 1

            self.costs[i] = tf.identity(self.costs[i] , name='cost_{}'.format(i))

            self.cost = tf.add(self.cost,self.costs[i],'combined_cost')
            summary.add_moving_summary(self.costs[i])

    def _get_optimizer(self):
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
