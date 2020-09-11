#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: tfutils.py
# Author: Qian Ge <geqian1001@gmail.com>
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def sample_normal_single(mean, stddev, name=None):
	return tf.random_normal(
		# shape=mean.get_shape(),
		shape=tf.shape(mean),
    	mean=mean,
    	stddev=stddev,
    	dtype=tf.float32,
    	seed=None,
    	name=name,
		)
