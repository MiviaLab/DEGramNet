import tensorflow as tf
import keras
from keras import backend as K
import numpy as np
import math
from keras.layers import Layer, Input, Conv1D, Dense, Reshape, Add, Lambda, GlobalMaxPooling1D, GlobalAveragePooling1D, Permute, multiply, Activation, Concatenate

#
# This module includes a Keras implementation of ECA layer described in
# "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks".
# https://arxiv.org/pdf/1910.03151.pdf
#

class EfficientChannelAttentionLayer(Layer):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, b=1, gamma=2, data_format="channels_last", dropout=0.0, **kwargs):
        super(EfficientChannelAttentionLayer, self).__init__(**kwargs)
        self.data_format = data_format
        self.channel_axis= -1 if self.data_format=="channels_last" else 1
        self.b = b
        self.gamma = gamma
        self.dropout_rate = dropout
    
    def build(self, input_shape):
        channels = input_shape[self.channel_axis]
        self.eca_k_size = int(np.ceil((np.log2(channels)+self.b/self.gamma) // 2 * 2 + 1))  #THIS IS NEXT NOT NEAREST
        
        self._global_avg_pool = GlobalAveragePooling1D(data_format=self.data_format)
        self._eca_conv = Conv1D(
            filters=1, 
            kernel_size=self.eca_k_size,
            padding='same', 
            use_bias=False,
            data_format='channels_first') 

        self._sigmoid = Activation('sigmoid')
        

    def call(self, x):

        # feature descriptor on the global spatial information

        # (B, 1, C)
        eca_features = K.expand_dims(self._global_avg_pool(x), axis=1)

        # (B, 1, C)
        eca_features_2 = self._eca_conv(eca_features)
       
        if self.data_format == "channels_first":
            # (B, C, 1)
            eca_features = K.permute_dimensions(eca_features, [0, 2, 1])

        # (B, C, 1)
        attn = self._sigmoid(eca_features)
        
        if K.learning_phase() == 1 and self.dropout_rate != 0.0 : #if training
          attn = tf.nn.dropout(attn, rate=self.dropout_rate)

        return multiply([x,attn])
        
        
        