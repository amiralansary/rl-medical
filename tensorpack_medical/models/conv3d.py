#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: conv3d.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
# Modified: Amir Alansary <amiralansary@gmail.com>

import tensorflow as tf
from tensorpack import layer_register, VariableHolder
from tensorpack.tfutils.common import get_tf_version_number
from .tflayer import rename_get_variable, convert_to_tflayer_args

from tensorpack_medical.utils.argtools import shape3d, shape5d, get_data_format3d


@layer_register(log_shape=True)
@convert_to_tflayer_args(
    args_names=['filters', 'kernel_size'],
    name_mapping={
        'out_channel': 'filters',
        'kernel_shape': 'kernel_size',
        'stride': 'strides',
    })

def Conv3D(
        inputs,
        filters,
        kernel_size,
        strides=(1, 1, 1),
        padding='same',
        data_format='channels_last',
        dilation_rate=(1, 1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        split=1):
    """
    A wrapper around `tf.layers.Conv3D`.
    Some differences to maintain backward-compatibility:
    1. Default kernel initializer is variance_scaling_initializer(2.0).
    2. Default padding is 'same'.
    3. Support 'split' argument to do group conv.
    Variable Names:
    * ``W``: weights
    * ``b``: bias
    """
    if split == 1:
        with rename_get_variable({'kernel': 'W', 'bias': 'b'}):
            layer = tf.layers.Conv3D(
                filters,
                kernel_size,
                strides=strides,
                padding=padding,
                data_format='channels_last',
                dilation_rate=dilation_rate,
                activation=activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer)
            ret = layer.apply(inputs, scope=tf.get_variable_scope())
            ret = tf.identity(ret, name='output')

        ret.variables = VariableHolder(W=layer.kernel)
        if use_bias:
            ret.variables.b = layer.bias

    else:
        # group conv implementation
        data_format = get_data_format3d(data_format, tfmode=False)
        in_shape = inputs.get_shape().as_list()
        channel_axis = 4 if data_format == 'NDHWC' else 1
        in_channel = in_shape[channel_axis]
        assert in_channel is not None, "[Conv3D] Input cannot have unknown channel!"
        assert in_channel % split == 0

        assert kernel_regularizer is None and bias_regularizer is None and activity_regularizer is None, \
            "Not supported by group conv now!"

        out_channel = filters
        assert out_channel % split == 0
        assert dilation_rate == (1, 1, 1) or get_tf_version_number() >= 1.5, 'TF>=1.5 required for group dilated conv'

        kernel_shape = shape3d(kernel_size)
        filter_shape = kernel_shape + [in_channel / split, out_channel]
        stride = shape5d(strides, data_format=data_format)

        kwargs = dict(data_format=data_format)
        if get_tf_version_number() >= 1.5:
            kwargs['dilations'] = shape4d(dilation_rate, data_format=data_format)

        W = tf.get_variable(
            'W', filter_shape, initializer=kernel_initializer)

        if use_bias:
            b = tf.get_variable('b', [out_channel], initializer=bias_initializer)

        inputs = tf.split(inputs, split, channel_axis)
        # tf.split(value,num_or_size_splits,axis=0, num=None,name='split')
        kernels = tf.split(W, split, 4)

        outputs = [tf.nn.conv3d(i, k, stride, padding.upper(), **kwargs)
                   for i, k in zip(inputs, kernels)]
        conv = tf.concat(outputs, channel_axis)
        if activation is None:
            activation = tf.identity
        ret = activation(tf.nn.bias_add(conv, b, data_format=data_format) if use_bias else conv, name='output')

        ret.variables = VariableHolder(W=W)
        if use_bias:
            ret.variables.b = b
    return ret

# @layer_register(log_shape=True)
# def Conv3D(x, out_channel, kernel_shape,
#            padding='SAME', stride=1,
#            W_init=None, b_init=None,
#            nl=tf.identity, split=1, use_bias=True,
#            data_format='NDHWC'):
#     """
#     3D convolution on 5D inputs.

#     Args:
#         x (tf.Tensor): a 5D tensor.
#             Must have known number of channels, but can have other unknown dimensions.
#         out_channel (int): number of output channel.
#         kernel_shape: (d, h, w) tuple or a int.
#         stride: (d, h, w) tuple or a int.
#         padding (str): 'valid' or 'same'. Case insensitive.
#         split (int): Split channels as used in Alexnet. Defaults to 1 (no split).
#         W_init: initializer for W. Defaults to `variance_scaling_initializer`.
#         b_init: initializer for b. Defaults to zero.
#         nl: a nonlinearity function.
#         use_bias (bool): whether to use bias.

#         data_format: An optional string from: "NDHWC", "NCDHW".
#             Defaults to "NDHWC". The data format of the input and output data.
#             With the default format "NDHWC", the data is stored in the order
#             of: [batch, in_depth, in_height, in_width, in_channels].
#             Alternatively, the format could be "NCDHW", the data storage order
#             is: [batch, in_channels, in_depth, in_height, in_width].

#     Returns:
#         tf.Tensor named ``output`` with attribute `variables`.

#     Variable Names:

#     * ``W``: weights
#     * ``b``: bias
#     """
#     in_shape = x.get_shape().as_list()
#     channel_axis = 4 if data_format == 'NDHWC' else 1
#     in_channel = in_shape[channel_axis]
#     assert in_channel is not None, "[Conv3D] Input cannot have unknown channel!"
#     assert in_channel % split == 0
#     assert out_channel % split == 0

#     kernel_shape = shape3d(kernel_shape)
#     padding = padding.upper()
#     filter_shape = kernel_shape + [in_channel / split, out_channel]
#     stride = shape5d(stride, data_format=data_format)

#     if W_init is None:
#         W_init = tf.contrib.layers.variance_scaling_initializer()
#     if b_init is None:
#         b_init = tf.constant_initializer()

#     W = tf.get_variable('W', filter_shape, initializer=W_init)

#     if use_bias:
#         b = tf.get_variable('b', [out_channel], initializer=b_init)

#     if split == 1:
#         conv = tf.nn.conv3d(x, W, stride, padding, data_format=data_format)
#     else:
#         inputs = tf.split(x, split, channel_axis)
#         kernels = tf.split(W, split, 3) # todo: this should be 3 or 4?
#         outputs = [tf.nn.conv3d(i, k, stride, padding, data_format=data_format)
#                    for i, k in zip(inputs, kernels)]
#         conv = tf.concat(outputs, channel_axis)

#     # todo: check data format in bias_add
#     ret = nl(tf.nn.bias_add(conv, b, data_format='NHWC') if use_bias else conv, name='output')
#     ret.variables = VariableHolder(W=W)
#     if use_bias:
#         ret.variables.b = b
#     return ret


@layer_register(log_shape=True)
def Deconv3D(x, out_shape, kernel_shape,
             stride, padding='SAME',
             W_init=None, b_init=None,
             nl=tf.identity, use_bias=True,
             data_format='NDHWC'):
    """
    3D deconvolution on 5D inputs.

    Args:
        x (tf.Tensor): a tensor of shape NDHWC.
            Must have known number of channels, but can have other unknown dimensions.
        out_shape: (d, h, w, channel) tuple, or just a integer channel,
            then (d, h, w) will be calculated by input_shape * stride
        kernel_shape: (d, h, w) tuple or a int.
        stride: (h, w) tuple or a int.
        padding (str): 'valid' or 'same'. Case insensitive.
        W_init: initializer for W. Defaults to `variance_scaling_initializer`.
        b_init: initializer for b. Defaults to zero.
        nl: a nonlinearity function.
        use_bias (bool): whether to use bias.

    Returns:
        tf.Tensor: a NDHWC tensor named ``output`` with attribute `variables`.

    Variable Names:

    * ``W``: weights
    * ``b``: bias
    """
    in_shape = x.get_shape().as_list()
    channel_axis = 4 if data_format == 'NDHWC' else 1
    in_channel = in_shape[channel_axis]
    assert in_channel is not None, "[Deconv3D] Input cannot have unknown channel!"
    kernel_shape = shape3d(kernel_shape)
    stride3d = shape3d(stride)
    stride5d = shape5d(stride, data_format=data_format)
    padding = padding.upper()
    in_shape_dyn = tf.shape(x)

    if isinstance(out_shape, int):
        out_channel = out_shape
        if data_format == 'NDHWC':
            shp3_0 = StaticDynamicAxis(in_shape[1], in_shape_dyn[1]).apply(lambda x: stride3d[0] * x)
            shp3_1 = StaticDynamicAxis(in_shape[2], in_shape_dyn[2]).apply(lambda x: stride3d[1] * x)
            shp3_2 = StaticDynamicAxis(in_shape[3], in_shape_dyn[3]).apply(lambda x: stride3d[2] * x)
            shp3_dyn = [shp3_0.dynamic, shp3_1.dynamic, shp3_2.dynamic, out_channel]
            shp3_static = [shp3_0.static, shp3_1.static, shp3_2.static, out_channel]
        else:
            shp3_0 = StaticDynamicAxis(in_shape[2], in_shape_dyn[2]).apply(lambda x: stride3d[0] * x)
            shp3_1 = StaticDynamicAxis(in_shape[3], in_shape_dyn[3]).apply(lambda x: stride3d[1] * x)
            shp3_2 = StaticDynamicAxis(in_shape[4], in_shape_dyn[4]).apply(lambda x: stride3d[2] * x)
            shp3_dyn = [out_channel, shp3_0.dynamic, shp3_1.dynamic, shp3_2.dynamic]
            shp3_static = [out_channel, shp3_0.static, shp3_1.static, shp3_2.static]
    else:
        for k in out_shape:
            if not isinstance(k, int):
                raise ValueError("[Deconv3D] out_shape {} is invalid!".format(k))
        out_channel = out_shape[channel_axis - 1] # out_shape doesn't have batch
        shp3_static = shp3_dyn = out_shape
    filter_shape = kernel_shape + [out_channel, in_channel]

    if W_init is None:
        W_init = tf.contrib.layers.variance_scaling_initializer() # xavier_initializer_conv2d()
    if b_init is None:
        b_init = tf.constant_initializer()
    W = tf.get_variable('W', filter_shape, initializer=W_init)
    if use_bias:
        b = tf.get_variable('b', [out_channel], initializer=b_init)

    out_shape_dyn = tf.stack([tf.shape(x)[0]] + shp3_dyn)
    conv = tf.nn.conv3d_transpose(
        x, W, out_shape_dyn, stride5d, padding=padding, data_format=data_format)
    conv.set_shape(tf.TensorShape([None] + shp3_static))
    ret = nl(tf.nn.bias_add(conv, b, data_format='NDHWC') if use_bias else conv, name='output')

    ret.variables = VariableHolder(W=W)
    if use_bias:
        ret.variables.b = b
    return ret
