#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import tensorflow as tf
from .detector import Detector

IMAGE_SIZE = 28
NUM_CLASSES = 2
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('train', 'train.txt', 'File name of train data')
flags.DEFINE_string('test', 'test.txt', 'File name of train data')
flags.DEFINE_string('train_dir', '/tmp/data', 'Directory to put the training data.')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 200, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 10, 'Batch size Must divide evenly into the dataset sizes.')


if __name__ == '__main__':
    def _load_files(filename):
        # ファイルを開く
        f = open(filename, 'r')
        # データを入れる配列
        images = []
        labels = []
        for line in f:
            # 改行を除いてスペース区切りにする
            line = line.rstrip()
            l = line.split()
            # データを読み込んで28x28に縮小
            img = cv2.imread(l[0])
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            # 一列にした後、0-1のfloat値にする
            images.append(img.flatten().astype(np.float32)/255.0)
            # ラベルを1-of-k方式で用意する
            tmp = np.zeros(NUM_CLASSES)
            tmp[int(l[1])] = 1
            labels.append(tmp)
        # numpy形式に変換
        images = np.asarray(images)
        labels = np.asarray(labels)
        f.close()
        return images, labels

    # 学習用画像をTensorFlowで読み込めるようTensor形式(行列)に変換
    train_image = []
    train_label = []
    train_image, train_label = _load_files(FLAGS.train)

    # 同じく検証用画像をTensorFlowで読み込めるようTensor形式(行列)に変換
    test_image = []
    test_label = []
    test_image, test_label = _load_files(FLAGS.test)

    with tf.Graph().as_default():
        detector = Detector()
        detector.train(
            train_image, train_lables, test_image, test_label,
            FLAGS.train_dir, FLAGS.learning_rate, FLAGS.max_steps, FLAGS.batch_size)
