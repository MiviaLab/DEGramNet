import tensorflow as tf
import keras
from keras import backend as K
import numpy as np
import math
from keras.layers import Layer, Input, Conv1D,  Lambda,  multiply
from keras.layers import Dense, Reshape, GlobalAveragePooling1D, Maximum, Add, Concatenate, Multiply 


#
# This module includes a Keras implementation of attention layers described in
# "Recalibrating Fully Convolutional Networks with Spatial and Channel ‘Squeeze & Excitation’ Blocks".
# https://arxiv.org/pdf/1808.08127.pdf


class SpatialSqueezeAndChannelExcitation(Layer):   # Classic SE block
    """
    implementation of https://arxiv.org/pdf/1709.01507.pdf
    refer to Spatial Squeeze and Channel Excitation Block (cSE) in https://arxiv.org/pdf/1808.08127.pdf
    """

    def __init__(self,ratio=0.5, data_format="channels_last", dropout=0.0, **kwargs):
        """
        """
        self.data_format = data_format
        self.ratio = ratio
        self.channel_axis = 1 if data_format == "channels_first" else -1
        self.dropout_rate = dropout

        super(SpatialSqueezeAndChannelExcitation, self).__init__(**kwargs)


    def build(self, input_shape):

        channel = input_shape[self.channel_axis]

        self._global_avg_pooling = GlobalAveragePooling1D(data_format= self.data_format)
        self._reshape = Reshape((1,channel))

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

        super(SpatialSqueezeAndChannelExcitation, self).build(input_shape)


    def call(self, x):

      se_feature = self._global_avg_pooling(x)
      se_feature = self._reshape(se_feature)
      se_feature = self._dense1(se_feature)
      se_feature = self._dense2(se_feature)

      if self.data_format =="channels_first":
        se_feature = K.permute_dimensions(se_feature, (0,2,1))
      
      if K.learning_phase() == 1 and self.dropout_rate != 0.0 : #if training
          se_feature = tf.nn.dropout(se_feature, rate=self.dropout_rate)
      
      return multiply([x, se_feature])

        
    def compute_output_shape(self, input_shape):
        return input_shape


class ChannelSqueezeAndSpatialExcitation(Layer):   # Classic SE block
    """
    refer to Channel Squeeze and Spatial Excitation Block (sSE) in https://arxiv.org/pdf/1808.08127.pdf
    """

    def __init__(self, data_format="channels_last", dropout=0.0, **kwargs):
        """
        """
        self.data_format = data_format
        self.dropout_rate = dropout

        super(ChannelSqueezeAndSpatialExcitation, self).__init__(**kwargs)


    def build(self, input_shape):

        self._channel_squeeze = Conv1D(filters=1, 
                                       kernel_size=1, 
                                       kernel_initializer="he_normal", 
                                       activation='sigmoid',
                                       data_format=self.data_format)
        self._channel_squeeze.build(input_shape)
        self.trainable_weights+=self._channel_squeeze.trainable_weights

        super(ChannelSqueezeAndSpatialExcitation, self).build(input_shape)


    def call(self, x):

      se_feature = self._channel_squeeze(x)
      
      if K.learning_phase() == 1 and self.dropout_rate != 0.0 : #if training
          se_feature = tf.nn.dropout(se_feature, rate=self.dropout_rate)
      
      return multiply([se_feature, x])

        
    def compute_output_shape(self, input_shape):
        return input_shape


class ChannelAndSpatialSqueezeAndExcitation(Layer):
    """
    refer to Channel Squeeze and Spatial Excitation Block (csSE) in https://arxiv.org/pdf/1808.08127.pdf
    """

    def __init__(self,ratio=0.5, mode='concurrent',strategy='maxout', data_format="channels_last", dropout=0.0, **kwargs):
        """
        N.B. strategy is ignored if mode is 'sequential'
        """
        if mode not in ['concurrent','sequential']:
          raise ValueError("mode must be one of ['concurrent','sequential']")
        
        self.data_format = data_format
        self.channel_axis = 1 if data_format == "channels_first" else -1
        self.ratio = ratio
        self.mode=mode
        self.strategy = strategy
        self.dropout_rate = dropout
        

        super(ChannelAndSpatialSqueezeAndExcitation, self).__init__(**kwargs)


    def build(self, input_shape):

        self._cSE = SpatialSqueezeAndChannelExcitation(ratio=self.ratio, data_format=self.data_format, dropout=self.dropout_rate)
        self._cSE.build(input_shape)
        self.trainable_weights+=self._cSE.trainable_weights

        self._sSE = ChannelSqueezeAndSpatialExcitation(data_format=self.data_format, dropout=self.dropout_rate)
        self._sSE.build(input_shape)
        self.trainable_weights+=self._sSE.trainable_weights

        self._strategy_layer =  self._get_strategy(self.strategy)
        
        super(ChannelAndSpatialSqueezeAndExcitation, self).build(input_shape)


    def call(self, x):

      cse_feature = self._cSE(x)

      if self.mode == 'concurrent':
        sse_feature = self._sSE(x)
        se_feature = self._strategy_layer([cse_feature,sse_feature])
      
      else:
        se_feature = self._sSE(cse_feature)

      
      return se_feature
    
    def _get_strategy(self, strategy):
      if strategy == "maxout":
        return Maximum()

      elif strategy == "add":
        return Add()

      elif strategy == "multiply":
        return Multiply()

      elif strategy == "concat":
        return Concatenate(axis=self.channel_axis)
      else:
        raise ValueError("strategy must be one of ['maxout','concat','multiply','add']")

        
    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        if self.mode=="concurrent" and self.strategy=="concat":
            input_shape[self.channel_axis]*=2
        return tuple(input_shape)
