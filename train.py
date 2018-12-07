#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import tensorflow as tf
from .model import DetectorModel

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('train', 'train.txt', 'File name of train data')
flags.DEFINE_string('test', 'test.txt', 'File name of train data')
flags.DEFINE_string('train_dir', '/tmp/data', 'Directory to put the training data.')
flags.DEFINE_integer('max_steps', 200, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 10, 'Batch size Must divide evenly into the dataset sizes.')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')


if __name__ == '__main__':
      # 学習用画像をTensorFlowで読み込めるようTensor形式(行列)に変換
      # ファイルを開く
      f = open(FLAGS.train, 'r')
      # データを入れる配列
      train_image = []
      train_label = []
      for line in f:
          # 改行を除いてスペース区切りにする
          line = line.rstrip()
          l = line.split()
          # データを読み込んで28x28に縮小
          img = cv2.imread(l[0])
          img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
          # 一列にした後、0-1のfloat値にする
          train_image.append(img.flatten().astype(np.float32)/255.0)
          # ラベルを1-of-k方式で用意する
          tmp = np.zeros(NUM_CLASSES)
          tmp[int(l[1])] = 1
          train_label.append(tmp)
        # numpy形式に変換
        train_image = np.asarray(train_image)
        train_label = np.asarray(train_label)
        f.close()

        # 同じく検証用画像をTensorFlowで読み込めるようTensor形式(行列)に変換
        f = open(FLAGS.test, 'r')
        test_image = []
        test_label = []
        for line in f:
            line = line.rstrip()
            l = line.split()
            img = cv2.imread(l[0])
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            test_image.append(img.flatten().astype(np.float32)/255.0)
            tmp = np.zeros(NUM_CLASSES)
            tmp[int(l[1])] = 1
            test_label.append(tmp)
        test_image = np.asarray(test_image)
        test_label = np.asarray(test_label)
        f.close()

    model = DetectorModel()

    with tf.Graph().as_default():
        # 画像を入れる仮のTensor
        images_placeholder = tf.placeholder("float", shape=(None, IMAGE_PIXELS))
        # ラベルを入れる仮のTensor
        labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))
        # dropout率を入れる仮のTensor
        keep_prob = tf.placeholder("float")

        # inference()を呼び出してモデルを作る
        logits = model.inference(images_placeholder, keep_prob)
        # loss()を呼び出して損失を計算
        loss_value = model.loss(logits, labels_placeholder)
        # training()を呼び出して訓練
        train_op = model.training(loss_value, FLAGS.learning_rate)
        # 精度の計算
        acc = detector.accuracy(logits, labels_placeholder)

        # 保存の準備
        saver = tf.train.Saver()
        # Sessionの作成
        sess = tf.Session()
        # 変数の初期化
        sess.run(tf.initialize_all_variables())
        # TensorBoardで表示する値の設定
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph_def)

        # 訓練の実行
        for step in range(FLAGS.max_steps):
            for i in range(len(train_image)/FLAGS.batch_size):
                # batch_size分の画像に対して訓練の実行
                batch = FLAGS.batch_size*i
                # feed_dictでplaceholderに入れるデータを指定する
                sess.run(train_op, feed_dict={
                    images_placeholder: train_image[batch:batch+FLAGS.batch_size],
                    labels_placeholder: train_label[batch:batch+FLAGS.batch_size],
                    keep_prob: 0.5})

                # 1 step終わるたびに精度を計算する
                train_accuracy = sess.run(acc, feed_dict={
                    images_placeholder: train_image,
                    labels_placeholder: train_label,
                    keep_prob: 1.0})
                print "step %d, training accuracy %g"%(step, train_accuracy)

                # 1 step終わるたびにTensorBoardに表示する値を追加する
                summary_str = sess.run(summary_op, feed_dict={
                    images_placeholder: train_image,
                    labels_placeholder: train_label,
                    keep_prob: 1.0})
                summary_writer.add_summary(summary_str, step)

        # 訓練が終了したらテストデータに対する精度を表示
        print "test accuracy %g"%sess.run(acc, feed_dict={
            images_placeholder: test_image,
            labels_placeholder: test_label,
            keep_prob: 1.0})

        # 最終的なモデルを保存
        save_path = saver.save(sess, "model.ckpt")
