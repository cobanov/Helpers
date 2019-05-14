# -*- coding: utf-8 -*-
"""
Created on Fri May  3 09:05:01 2019

@author: COB3BU
"""

import tensorflow as tf

hello = tf.constant("hello world")

sess = tf.Session()

print(sess.run(hello))