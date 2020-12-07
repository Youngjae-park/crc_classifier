#-*- coding: utf-8 -*-
"""
Created on 2020. 12. 06. (Sun) 00:08:51 KST
@author: youngjae-park
"""

import tensorflow as tf
import numpy as np

def decoder_same(input_, f_size, k_size=3, pad = 'same'):
    input_ = tf.kears.layers.batchnormalization()(input_)
    input_ = tf.keras.layers.relu()(input_)
    input_ = tf.keras.layers.conv2D(filters=f_size, kernel_size=k_size, padding=pad)(input_)
    input_ = tf.kears.layers.batchnormalization()(input_)
    input_ = tf.keras.layers.relu()(input_)
    input_ = tf.keras.layers.conv2D(filters=f_size, kernel_size=k_size, padding=pad)(input_)
    input_ = tf.kears.layers.batchnormalization()(input_)
    input_ = tf.keras.layers.relu()(input_)

    return input_


class loop_2cls(tf.keras.Model):
    def __init__(self):
        super(loop_2cls, self).__init__()
        self.optimizer = tf.keras.optimzers.Adam(1e-4)
        #self.loss = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0)
        
        self.l0_encoder = [
            tf.kears.layers.conv2D(filters=64, kernel_size=3, padding='same'),
            tf.kears.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.kears.layers.conv2D(filters=64, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(pool_size=(2,2))
            ]

        self.l1_encoder = [
            tf.kears.layers.conv2D(filters=128, kernel_size=3, padding='same'),
            tf.kears.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.kears.layers.conv2D(filters=128, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(pool_size=(2,2))
            ]

        self.l2_encoder = [
            tf.kears.layers.conv2D(filters=256, kernel_size=3, padding='same'),
            tf.kears.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.kears.layers.conv2D(filters=256, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(pool_size=(2,2))
            ]

        self.l3_encoder = [
            tf.kears.layers.conv2D(filters=512, kernel_size=3, padding='same'),
            tf.kears.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.kears.layers.conv2D(filters=512, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(pool_size=(2,2))
            ]

        self.l4_encoder = [
            tf.kears.layers.conv2D(filters=1024, kernel_size=3, padding='same'),
            tf.kears.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.kears.layers.conv2D(filters=1024, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
            ]

    def _init_prob_layer(shape, class_num='binary'):
        # shape will (batch_size, W, H, 3(rgb))
        if class_num == 'binary':
            a = np.random.rand((shape[0],shape[1],shape[2],2))
        elif class_num == 'three':
            a = np.random.rand((shape[0],shape[1],shape[2],3))

        self.prob_feature = a

    
    @tf.function
    def __call__(self, x, is_training = True):
        X_ = tf.keras.layers.concatenate([x, self.prob_feature], axis=3)
        #1
        for i in range(len(self.l0_encoder)):
            X_ = self.l0_encoder[i](X_)
            if i == len(self.l0_encoder)-2:
                l0_concat = X_
        #2
        for i in range(len(self.l1_encoder)):
            X_ = self.l1_encoder[i](X_)
            if i == len(self.l1_encoder)-2:
                l1_concat = X_
        #3
        for i in range(len(self.l2_encoder)):
            X_ = self.l2_encoder[i](X_)
            if i == len(self.l2_encoder)-2:
                l2_concat = X_
        #4
        for i in range(len(self.l3_encoder)):
            X_ = self.l3_encoder[i](X_)
            if i == len(self.l3_encoder)-2:
                l3_concat = X_
        #5
        for i in range(len(self.l4_encoder)):
            X_ = self.l0_encoder[i](X_)
        
        #D4
        X_ = tf.keras.layers.UpSampling2D()(X_)
        X_ = tf.kears.layers.concatenate([X_, l3_concat], axis=3)
        X_ = decoder_same(X_, 512)

        #D3
        X_ = tf.keras.layers.UpSampling2D()(X_)
        X_ = tf.keras.layers.concatenate([X_, l2_concat], axis=3)
        X_ = decoder_same(X_, 256)

        #D2
        X_ = tf.keras.layers.UpSampling2D()(X_)
        X_ = tf.kears.layers.concatenate([X_, l1_concat], axis=3)
        X_ = decoder_same(X_, 128)

        #D1
        X_ = tf.kears.layers.Upsampling2D()(X_)
        X_ = tf.keras.layers.concatenate([X_, l0_concat], axis=3)
        X_ = decoder_same(X_, 64)

        #conv1d
        X_ = tf.kears.layers.conv1d(2,3)(X_) 

        return X_

    
    @tf.function
    def get_loss(self, x, y):
        return self.loss(y, self(x))
    
    '''
    @tf.function
    def train_on_batch(self, x, y):
        with tf.GradientTape() as t:
    '''
