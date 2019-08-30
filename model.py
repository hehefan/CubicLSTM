import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from lstm_ops import cubic_lstm_cell

import tensorflow as tf

tf.app.flags.DEFINE_integer("kernel_channels", 32, "Output channels for conv kernels.")
tf.app.flags.DEFINE_integer("kernel_shape_x", 3, "Spatial Conv kernel shape of CubicLSTM.")
tf.app.flags.DEFINE_integer("kernel_shape_y", 1, "Temporal Conv kernel shape of CubicLSTM.")
tf.app.flags.DEFINE_integer("kernel_shape_z", 3, "Output Conv kernel shape of CubicLSTM.")
tf.app.flags.DEFINE_integer("kernel_shape_o", 5, "Output Conv kernel shape of model.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of RNN layers.")
tf.app.flags.DEFINE_integer("num_wins", 3, "Size of windows.")

tf.app.flags.DEFINE_float("learning_rate", 1e-4, "Learning rate.")
tf.app.flags.DEFINE_integer("num_gpus", 8, "Number of GPUs.")

FLAGS = tf.app.flags.FLAGS

def tower_loss(scope, batch_size, video, num_layers=FLAGS.num_layers, num_wins=FLAGS.num_wins, filter_size_x=FLAGS.kernel_shape_x, filter_size_y=FLAGS.kernel_shape_y, filter_size_z=FLAGS.kernel_shape_z, filter_size_o=FLAGS.kernel_shape_o, kernel_channels=FLAGS.kernel_channels):
  images = tf.split(value=video, num_or_size_splits=20, axis=3)

  input_images = [None] * num_wins
  states_x = [[None]*num_layers] * num_wins
  states_y = [None]*num_layers

  for i in range(num_wins):
    input_images[i] = images[i:10-num_wins+i+1]

  for i in range(10-num_wins+1):    # step
    with tf.variable_scope("Encoder", reuse=(i>0)):
      for j in range(num_wins):       # window
        output = input_images[j][i]
        for k in range(num_layers):   # layer
          output, states_x[j][k], states_y[k] = cubic_lstm_cell(output, states_x[j][k], states_y[k], num_channels=kernel_channels, filter_size_x=filter_size_x, filter_size_y=filter_size_y, filter_size_z=filter_size_z, scope="%d_%d"%(j,k))

  outputs = images[10-num_wins:10]

  for i in range(10):
    with tf.variable_scope("Decoder", reuse=(i>0)):
      for j in range(num_wins):
        output = outputs[-num_wins]
        for k in range(num_layers):   # layer
          output, states_x[j][k], states_y[k] = cubic_lstm_cell(output, states_x[j][k], states_y[k], num_channels=kernel_channels, filter_size_x=filter_size_x, filter_size_y=filter_size_y, filter_size_z=filter_size_z, scope="%d_%d"%(j,k))
      output = slim.layers.conv2d(output, 1, [filter_size_o, filter_size_o], stride=1, activation_fn=None, scope='Output')
      outputs.append(output)

  outputs = outputs[num_wins:]

  ce = 0.0 
  for output, image in zip(outputs, images[10:]):
    ce += tf.nn.sigmoid_cross_entropy_with_logits(labels=image, logits=output)
  ce = tf.reduce_mean(ce)/10.0
  return ce

def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:

      if g is None:
        continue

      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    if len(grads) > 0:
      # Average over the 'tower' dimension.
      grad = tf.concat(axis=0, values=grads)
      grad = tf.reduce_mean(grad, 0)

      # Keep in mind that the Variables are redundant because they are shared
      # across towers. So .. we will just return the first tower's pointer to
      # the Variable.
      v = grad_and_vars[0][1]
      grad_and_var = (grad, v)
      average_grads.append(grad_and_var)
  return average_grads

class Model(object):
  def __init__(self, batch_size, learning_rate=FLAGS.learning_rate):

    with tf.device('/cpu:0'):

      self.video_ph = tf.placeholder(tf.float32, [batch_size*FLAGS.num_gpus, 64, 64, 20])
      batch_videos = tf.split(value=self.video_ph, num_or_size_splits=FLAGS.num_gpus, axis=0)

      # Create an optimizer that performs gradient descent.
      opt = tf.train.AdamOptimizer(learning_rate)

      tower_grads, total_loss = [], 0.0
      with tf.variable_scope(tf.get_variable_scope()):
        for i in range(FLAGS.num_gpus):
          with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:
              # all towers.
              loss = tower_loss(scope=scope, batch_size=batch_size, video=batch_videos[i])
              total_loss += loss
              # Reuse variables for the next tower.
              tf.get_variable_scope().reuse_variables()
              # Calculate the gradients for the batch of data on this CIFAR tower.
              grads = opt.compute_gradients(loss)
              # Keep track of the gradients across all towers.
              tower_grads.append(grads)

      # We must calculate the mean of each gradient. Note that this is the
      # synchronization point across all towers.
      grads = average_gradients(tower_grads)

      # Apply the gradients to adjust the shared variables.
      self.train_op = opt.apply_gradients(grads)

      self.ce = total_loss/FLAGS.num_gpus

      # Create a saver.
      self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=0)
