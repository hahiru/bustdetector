#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import tensorflow as tf


class DetectorModel:
    __slots__ = ['_num_class', '_image_size', '_number_of_color', '_image_pixels']
    
    def __init__(self, num_class, image_size, number_of_color):
        self._num_class = num_class
        self._image_size = image_size
        self._number_of_color = number_of_color
        self._image_pixels = image_size*image_size*number_of_color
    
    def inference(self, images_placeholder, keep_prob):
        x_image = tf.reshape(images_placeholder, [-1, self._image_size, self._image_size, self._number_of_color])

        with tf.name_scope('conv1') as scope:
            W_conv1 = self._weight_valiable([5, 5, 1, 32])
            b_conv1 = self._bias_variable([32])
            h_conv1 = tf.nn.relu(self._conv2d(x_image, W_conv1) + b_conv1)

        with tf.name_scope('pool1') as scope:
            h_pool1 = self._max_pool_2x2(h_conv1)

        with tf.name_scope('conv2') as scope:
            W_conv2 = self._weight_valiable([5, 5, 32, 64])
            b_conv2 = self._bias_variable([64])
            h_conv2 = tf.nn.relu(self._conv2d(h_pool1, W_conv2) + b_conv2)

        with tf.name_scope('pool2') as scope:
            h_pool2 = self._max_pool_2x2(h_conv2)

        with tf.name_scope('fc1') as scope:
            W_fc1 = self._weight_valiable([7*7*64, 1024])
            b_fc1 = self._bias_variable([1024])
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
            # dropoutの設定
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        with tf.name_scope('fc2') as scope:
            W_fc2 = self._weight_valiable([1024, self._num_class])
            b_fc2 = self._bias_variable([self._num_class])

        with tf.name_scope('softmax') as scope:
            y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        return y_conv

    # 重みを標準偏差0.1の正規分布で初期化
    def _weight_valiable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # バイアスを標準偏差0.1の正規分布で初期化
    def _bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def _conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def _max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def loss(self, logits, labels):
        cross_entropy = -tf.reduce_sum(labels*tf.log(logits))
        tf.summary.scalar("cross_entropy", cross_entropy)
        return cross_entropy

    def training(self, loss, learning_rate):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return train_step

    def accuracy(self, logits, labels):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        tf.summary.scalar("accuracy", accuracy)
        return accuracy
