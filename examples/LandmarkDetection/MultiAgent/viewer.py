#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: viewer.py
# Author: Amir Alansary <amiralansary@gmail.com>

import os
import math
import pyglet
from pyglet.gl import (
    gl,
    glTexParameteri,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_LINEAR,
    GL_TEXTURE_MIN_FILTER,
    glScalef,
    glEnable,
    GL_BLEND,
    glBlendFunc,
    GL_SRC_ALPHA,
    GL_ONE_MINUS_SRC_ALPHA,
    GLubyte,
    glBegin,
    GL_POINTS,
    glVertex3f,
    glEnd,
    GL_QUADS,
    GL_POLYGON,
    GL_TRIANGLES,
    glColor4f)


class SimpleImageViewer(object):
    ''' Simple image viewer class for rendering images using pyglet'''

    def __init__(self, arr, scale_x=1, scale_y=1, filepath=None, display=None):

        self.isopen = False
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.display = display
        self.filepath = filepath
        self.filename = os.path.basename(filepath)

        # initialize window with the input image
        height, width, channels = arr.shape
        assert arr.shape == (
            height, width, 3), """You passed in an image with the wrong number
            shape"""
        self.window = pyglet.window.Window(width=scale_x * width,
                                           height=scale_y * height,
                                           caption=self.filename,
                                           display=self.display,
                                           resizable=True,
                                           # fullscreen=True # ruins screen
                                           # resolution
                                           )

        # set location
        # screen_width = self.window.display.get_default_screen().width
        # screen_height = self.window.display.get_default_screen().height
        self.location_x = 0  # screen_width / 2 #- 2* width
        self.location_y = 50  # screen_height / 2 #- 2* height
        self.window.set_location(
            (int)(
                self.location_x), (int)(
                self.location_y))

        # scale window size
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glScalef(scale_x, scale_y, 1.0)

        self.img_width = width
        self.img_height = height
        self.isopen = True

        self.window_width, self.window_height = self.window.get_size()

        # turn on transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def draw_image(self, arr):
        # convert data typoe to GLubyte
        rawData = (GLubyte * arr.size)(*list(arr.ravel().astype('int')))
        image = pyglet.image.ImageData(self.img_width, self.img_height, 'RGB',
                                       rawData,  # arr.tostring(),
                                       pitch=self.img_width * -3)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        image.blit(0, 0)

    def draw_point(self, x=0.0, y=0.0, z=0.0):
        x = self.img_height - x
        y = y
        # pyglet.graphics.draw(1, GL_POINTS,
        #     ('v2i', (x_new, y_new)),
        #     ('c3B', (255, 0, 0))
        # )
        glBegin(GL_POINTS)  # draw point
        glVertex3f(x, y, z)
        glEnd()

    def draw_circle(self, radius=10, res=30, pos_x=0, pos_y=0,
                    color=(1.0, 1.0, 1.0, 1.0), **attrs):

        points = []
        # window start indexing from bottom left
        x = self.img_height - pos_x
        y = pos_y

        for i in range(res):
            ang = 2 * math.pi * i / res
            points.append((math.cos(ang) * radius + y,
                           math.sin(ang) * radius + x))

        # draw filled polygon
        if len(points) == 4:
            glBegin(GL_QUADS)
        elif len(points) > 4:
            glBegin(GL_POLYGON)
        else:
            glBegin(GL_TRIANGLES)
        for p in points:
            # choose color
            glColor4f(color[0], color[1], color[2], color[3])
            glVertex3f(p[0], p[1], 0)  # draw each vertex
        glEnd()
        # reset color
        glColor4f(1.0, 1.0, 1.0, 1.0)

    def draw_rect(self, x_min_init, y_min, x_max_init, y_max):
        main_batch = pyglet.graphics.Batch()
        # fix location
        x_max = self.img_height - x_max_init
        x_min = self.img_height - x_min_init
        # draw lines
        glColor4f(0.8, 0.8, 0.0, 1.0)
        main_batch.add(2, gl.GL_LINES, None,
                       ('v2f', (y_min, x_min, y_max, x_min)))
        # ('c3B', (204, 204, 0, 0, 255, 0)))
        main_batch.add(2, gl.GL_LINES, None,
                       ('v2f', (y_min, x_min, y_min, x_max)))
        # ('c3B', (204, 204, 0, 0, 255, 0)))
        main_batch.add(2, gl.GL_LINES, None,
                       ('v2f', (y_max, x_max, y_min, x_max)))
        # ('c3B', (204, 204, 0, 0, 255, 0)))
        main_batch.add(2, gl.GL_LINES, None,
                       ('v2f', (y_max, x_max, y_max, x_min)))
        # ('c3B', (204, 204, 0, 0, 255, 0)))

        main_batch.draw()
        # reset color
        glColor4f(1.0, 1.0, 1.0, 1.0)

    def display_text(self, text, x, y, color=(0, 0, 204, 255),  # RGBA
                     anchor_x='left', anchor_y='top'):
        x = int(self.img_height - x)
        y = int(y)
        label = pyglet.text.Label(text,
                                  font_name='Ariel', color=color,
                                  font_size=8, bold=True,
                                  x=y, y=x,
                                  anchor_x=anchor_x, anchor_y=anchor_y)
        label.draw()

    def render(self):
        self.window.flip()

    def saveGif(self, filename=None, arr=None, duration=0):
        arr[0].save(filename, save_all=True,
                    append_images=arr[1:],
                    duration=500,
                    quality=95)  # duration milliseconds

    def close(self):
        if self.isopen:
            self.window.close()
            self.isopen = False

    def __del__(self):
        self.close()
