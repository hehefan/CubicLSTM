"""Cubic Convolutional LSTM implementation."""

import tensorflow as tf

from tensorflow.contrib.slim import add_arg_scope
from tensorflow.contrib.slim import layers

def init_state(inputs,
               state_shape,
               state_initializer=tf.zeros_initializer(),
               dtype=tf.float32):
  """Helper function to create an initial state given inputs.

  Args:
    inputs: input Tensor, at least 2D, the first dimension being batch_size
    state_shape: the shape of the state.
    state_initializer: Initializer(shape, dtype) for state Tensor.
    dtype: Optional dtype, needed when inputs is None.
  Returns:
     A tensors representing the initial state.
  """
  if inputs is not None:
    # Handle both the dynamic shape as well as the inferred shape.
    inferred_batch_size = inputs.get_shape().with_rank_at_least(1)[0]
    dtype = inputs.dtype
  else:
    inferred_batch_size = 0
  initial_state = state_initializer(
      [inferred_batch_size] + state_shape, dtype=dtype)
  return initial_state


@add_arg_scope
def cubic_lstm_cell(inputs,
                    state_x,
                    state_y,
                    num_channels,
                    filter_size_x=3,
                    filter_size_y=1,
                    filter_size_z=5,
                    forget_bias=1.0,
                    scope=None,
                    reuse=None):

  spatial_size = inputs.get_shape()[1:3] 
  if state_x is None:
    state_x = init_state(inputs, list(spatial_size) + [2 * num_channels])
  if state_y is None:
    state_y = init_state(inputs, list(spatial_size) + [2 * num_channels])
  with tf.variable_scope(scope, 'CubicLstmCell', [inputs, state_x, state_y], reuse=reuse):
    inputs.get_shape().assert_has_rank(4)
    state_x.get_shape().assert_has_rank(4)
    state_y.get_shape().assert_has_rank(4)

    c_x, h_x = tf.split(axis=3, num_or_size_splits=2, value=state_x)
    c_y, h_y = tf.split(axis=3, num_or_size_splits=2, value=state_y)

    inputs_h = tf.concat(axis=3, values=[inputs, h_x, h_y])

    # Spatial 
    i_j_f_o_y = layers.conv2d(inputs_h,
                              4 * num_channels, [filter_size_x, filter_size_x],
                              stride=1,
                              activation_fn=None,
                              scope='GatesY')

    i_y, j_y, f_y, o_y = tf.split(axis=3, num_or_size_splits=4, value=i_j_f_o_y)

    new_c_y = c_y * tf.sigmoid(f_y + forget_bias) + tf.sigmoid(i_y) * tf.tanh(j_y)
    new_h_y = tf.tanh(new_c_y) * tf.sigmoid(o_y)

    # Temporal
    i_j_f_o_x = layers.conv2d(inputs_h,
                              4 * num_channels, [filter_size_y, filter_size_y],
                              stride=1,
                              activation_fn=None,
                              scope='GatesX')

    i_x, j_x, f_x, o_x = tf.split(axis=3, num_or_size_splits=4, value=i_j_f_o_x)

    new_c_x = c_x * tf.sigmoid(f_x + forget_bias) + tf.sigmoid(i_x) * tf.tanh(j_x)
    new_h_x = tf.tanh(new_c_x) * tf.sigmoid(o_x)

    # Output
    new_h = layers.conv2d(tf.concat(axis=3, values=[new_h_x, new_h_y]),
                              num_channels, [filter_size_z, filter_size_z],
                              stride=1,
                              activation_fn=None,
                              scope='Output')

    return new_h, tf.concat(axis=3, values=[new_c_x, new_h_x]), tf.concat(axis=3, values=[new_c_y, new_h_y])
