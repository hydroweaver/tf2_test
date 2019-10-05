# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 20:33:53 2019

@author: hydro
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255, x_test / 255