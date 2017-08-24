#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: argtools.py


import inspect
import six
from tensorpack import logger
if six.PY2:
    import functools32 as functools
else:
    import functools

__all__ = ['shape3d', 'shape5d']



def shape3d(a):
    """
    Ensure a 3D shape.

    Args:
        a: a int or tuple/list of length 3

    Returns:
        list: of length 3. if ``a`` is a int, return ``[a, a, a]``.
    """
    if type(a) == int:
        return [a, a, a]
    if isinstance(a, (list, tuple)):
        assert len(a) == 3
        return list(a)
    raise RuntimeError("Illegal shape: {}".format(a))


def shape5d(a, data_format='NDHWC'):
    """
    Ensuer a 5D shape, to use with 5D symbolic functions.

    Args:
        a: a int or tuple/list of length 3

    Returns:
        list: of length 5. if ``a`` is a int, return ``[1, a, a, a, 1]``
            or ``[1, 1, a, a, a]`` depending on data_format "NDHWC" or "NCDHW".
    """
    s2d = shape3d(a)
    if data_format == 'NDHWC':
        return [1] + s2d + [1]
    else:
        return [1, 1] + s2d
