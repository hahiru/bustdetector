#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
from .model import DetectorModel

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*3


class BustDetector:
    __slots__ = ['_sess', '_model']

    def __init__(self):
        self._sess = tf.Session()
        self._model = DetectorModel()

    def train(self, train_image, train_label, test_image, test_label,
              train_dir, learning_rate, max_steps, batch_size):
        # 画像を入れる仮のTensor
        images_placeholder = tf.placeholder("float", shape=(None, IMAGE_PIXELS))
        # ラベルを入れる仮のTensor
        labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))
        # dropout率を入れる仮のTensor
        keep_prob = tf.placeholder("float")

        # inference()を呼び出してモデルを作る
        logits = self._model.inference(images_placeholder, keep_prob)
        # loss()を呼び出して損失を計算
        loss_value = self._model.loss(logits, labels_placeholder)
        # training()を呼び出して訓練
        train_op = self._model.training(loss_value, learning_rate)
        # 精度の計算
        acc = detector.accuracy(logits, labels_placeholder)

        # 保存の準備
        saver = tf.train.Saver()
        # 変数の初期化
        self._sess.run(tf.initialize_all_variables())
        # TensorBoardで表示する値の設定
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(train_dir, sess.graph_def)

        # 訓練の実行
        for step in range(max_steps):
            for i in range(len(train_image)/batch_size):
                # batch_size分の画像に対して訓練の実行
                batch = batch_size * i
                # feed_dictでplaceholderに入れるデータを指定する
                self._sess.run(train_op, feed_dict={
                    images_placeholder: train_image[batch:batch+batch_size],
                    labels_placeholder: train_label[batch:batch+batch_size],
                    keep_prob: 0.5})

                # 1 step終わるたびに精度を計算する
                train_accuracy = self._sess.run(acc, feed_dict={
                    images_placeholder: train_image,
                    labels_placeholder: train_label,
                    keep_prob: 1.0})
                print "step %d, training accuracy %g"%(step, train_accuracy)

                # 1 step終わるたびにTensorBoardに表示する値を追加する
                summary_str = self._sess.run(summary_op, feed_dict={
                    images_placeholder: train_image,
                    labels_placeholder: train_label,
                    keep_prob: 1.0})
                summary_writer.add_summary(summary_str, step)

        # 訓練が終了したらテストデータに対する精度を表示
        print "test accuracy %g"%self._sess.run(acc, feed_dict={
            images_placeholder: test_image,
            labels_placeholder: test_label,
            keep_prob: 1.0})

        # 最終的なモデルを保存
        save_path = saver.save(self._sess, "model.ckpt")
