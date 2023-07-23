
import tensorflow as tf
from keras.layers import Layer

class ZScoreNormalization(Layer):
  def __init__(self, axis=[-1,-2] , scale_variance=True, eps=1e-07,**kwargs):
    self.axis=axis
    self.scale = scale_variance
    self.eps = eps
    super(ZScoreNormalization,self).__init__(**kwargs)

  def build(self,input_shape):
    super(ZScoreNormalization,self).build(input_shape)
  
  def call(self, input_tensor):
    mean_values = tf.math.reduce_mean(input_tensor,axis=self.axis,keepdims=True)

    if self.scale:
      dev_std = tf.math.reduce_std(input_tensor,axis=self.axis,keepdims=True) + tf.constant(self.eps)
      norm_tensor = (input_tensor -mean_values)/dev_std
    else:
      norm_tensor = input_tensor -mean_values
      
    return norm_tensor
  
  def compute_output_shape(self, input_shape):
    return input_shape