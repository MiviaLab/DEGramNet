import tensorflow as tf
import keras
from keras import backend as K
from keras.utils import conv_utils
import numpy as np
import math
from keras import initializers
from keras.layers import Layer, Input, Conv1D, Dense, Reshape, Add, Lambda, GlobalMaxPooling1D, GlobalAveragePooling1D, Permute, multiply, Activation, Concatenate, TimeDistributed, Conv2D

import os

#
# This module includes a Keras implementation of Squeeze-And_Excite block described in
# "Squeeze-and-Excitation Networks".
# https://arxiv.org/pdf/1709.01507.pdf


class SqueezeAndExciteLayer(Layer):

    def __init__(self, ratio=0.5, data_format="channels_last", dropout=0.0, **kwargs):
        """
        """
        self.data_format = data_format
        self.ratio = ratio
        self.channel_axis = 1 if data_format == "channels_first" else -1
        self.dropout_rate = dropout

        super(SqueezeAndExciteLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        channel = input_shape[self.channel_axis]

        self._global_avg_pooling = GlobalAveragePooling1D(
            data_format=self.data_format)
        #self._reshape = Reshape((1, channel))

        self._dense1 = Dense(int(channel*self.ratio),
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
        self._dense1.build((1, 1, channel))
        self.trainable_weights += self._dense1.trainable_weights

        self._dense2 = Dense(channel,
                             activation='sigmoid',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
        out_dense1_shape = self._dense1.compute_output_shape((1, 1, channel))
        self._dense2.build(out_dense1_shape)
        self.trainable_weights += self._dense2.trainable_weights

        super(SqueezeAndExciteLayer, self).build(input_shape)

    def call(self, x):

        se_feature = self._global_avg_pooling(x)
        #se_feature = self._reshape(se_feature)
        se_feature = self._dense1(se_feature)
        se_feature = self._dense2(se_feature)

        if self.data_format == "channels_first":
          se_feature = K.expand_dims(se_feature, axis=-1)
        else:
          se_feature = K.expand_dims(se_feature, axis=1)
        
        if K.learning_phase() == 1 and self.dropout_rate != 0.0 : #if training
          se_feature = tf.nn.dropout(se_feature, rate=self.dropout_rate)

        return multiply([x, se_feature])

    def compute_output_shape(self, input_shape):
        return input_shape

