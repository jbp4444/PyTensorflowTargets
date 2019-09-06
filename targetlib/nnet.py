#
# original code:  Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""was: A deep MNIST classifier using convolutional layers.
   now: A deep CNN for finding x,y center of a block
      dc3 includes rotation prediction (1 of 4, NSEW)

"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

import tensorflow as tf

import targetlib

def create_deepnn( sess ):
	"""deepnn builds the graph for a deep net for classifying digits.

	Args:
	x: an input tensor with the dimensions (N_examples, 784), where 784 is the
	number of pixels in a standard MNIST image.

	Returns:
	A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
	equal to the logits of classifying the digit into one of 10 classes (the
	digits 0-9). keep_prob is a scalar placeholder for the probability of
	dropout.
	"""
	img = tf.placeholder(tf.float32, [None, 784], name='img')

	# Define loss and optimizer
	xc = tf.placeholder(tf.float32, [None, 28], name='xc' )  # was [None, 10]
	yc = tf.placeholder(tf.float32, [None, 28], name='yc' )  # was [None, 10]
	rc = tf.placeholder(tf.float32, [None, 4], name='rc' )  # was [None, 10]

	keep_prob = tf.placeholder( tf.float32, name='keep_prob' )

	# Reshape to use within a convolutional neural net.
	# Last dimension is for "features" - there is only one here, since images are
	# grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
	with tf.name_scope('xycommon'):
		x_image = tf.reshape( img, [-1, 28, 28, 1], name='x_image' )

		# First convolutional layer - maps one grayscale image to 32 feature maps.
		W_conv1 = weight_variable([5, 5, 1, 32])
		b_conv1 = bias_variable([32])
		h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

		# Pooling layer - downsamples by 2X.
		h_pool1 = max_pool_2x2(h_conv1)

		# Second convolutional layer -- maps 32 feature maps to 64.
		W_conv2 = weight_variable([5, 5, 32, 64])
		b_conv2 = bias_variable([64])
		h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

		# Second pooling layer.
		h_pool2 = max_pool_2x2(h_conv2)
		h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

	# Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
	# is down to 7x7x64 feature maps -- maps this to 1024 features.
	with tf.name_scope('xcoord'):
		W_fc1 = weight_variable([7 * 7 * 64, 1024])
		b_fc1 = bias_variable([1024])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

		W_fc2 = weight_variable([1024, 28])  # was [1024,10]
		b_fc2 = bias_variable([28])  # was [10]
		xc_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	# ADDING A SECOND SET OF DNN nodes
	# Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
	# is down to 7x7x64 feature maps -- maps this to 1024 features.
	with tf.name_scope('ycoord'):
		W_fc3 = weight_variable([7 * 7 * 64, 1024])
		b_fc3 = bias_variable([1024])
		h_fc3 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc3) + b_fc3)

		h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob )

		W_fc4 = weight_variable([1024, 28])  # was [1024,10]
		b_fc4 = bias_variable([28])  # was [10]
		yc_conv = tf.matmul(h_fc3_drop, W_fc4) + b_fc4

	# ADDING A THIRD SET OF DNN nodes
	# Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
	# is down to 7x7x64 feature maps -- maps this to 1024 features.
	with tf.name_scope('rotation'):
		#W_conv1b = weight_variable([5, 5, 1, 32])
		#b_conv1b = bias_variable([32])
		W_conv1b = weight_variable([7, 7, 1, 64])
		b_conv1b = bias_variable([64])
		h_conv1b = tf.nn.relu(conv2d(x_image, W_conv1b) + b_conv1b)

		# Pooling layer - downsamples by 2X.
		h_pool1b = max_pool_2x2(h_conv1b)

		# Second convolutional layer -- maps 32 feature maps to 64.
		#W_conv2b = weight_variable([5, 5, 32, 64])
		#b_conv2b = bias_variable([64])
		W_conv2b = weight_variable([5, 5, 64, 128])
		b_conv2b = bias_variable([128])
		h_conv2b = tf.nn.relu(conv2d(h_pool1b, W_conv2b) + b_conv2b)

		# Second pooling layer.
		h_pool2b = max_pool_2x2(h_conv2b)
		#h_pool2b_flat = tf.reshape(h_pool2b, [-1, 7*7*64])
		h_pool2b_flat = tf.reshape(h_pool2b, [-1, 7*7*128])

		#W_fc5 = weight_variable([7 * 7 * 64, 1024])
		W_fc5 = weight_variable([7 * 7 * 128, 1024])
		b_fc5 = bias_variable([1024])
		h_fc5 = tf.nn.relu(tf.matmul(h_pool2b_flat, W_fc5) + b_fc5)

		h_fc5_drop = tf.nn.dropout(h_fc5, keep_prob )

		W_fc6 = weight_variable([1024, 4])  # was [1024,10]
		b_fc6 = bias_variable([4])  # was [10]
		rc_conv = tf.matmul(h_fc5_drop, W_fc6) + b_fc6

	with tf.name_scope('loss'):
		xc_loss = tf.nn.softmax_cross_entropy_with_logits(
				labels = xc, logits=xc_conv )
		yc_loss = tf.nn.softmax_cross_entropy_with_logits(
				labels = yc, logits=yc_conv )
		rc_loss = tf.nn.softmax_cross_entropy_with_logits(
				labels = rc, logits=rc_conv )
		sumXY_loss = xc_loss + yc_loss
		#joint_loss = xc_loss + yc_loss + 40*rc_loss
		joint_loss = sumXY_loss + 100*rc_loss
		#joint_loss = rc_loss

	with tf.name_scope('adam_optimizer'):
		#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
		#train_step = tf.train.AdamOptimizer(1e-4).minimize(joint_loss)
		train_stepXY = tf.train.AdamOptimizer(1e-5).minimize(sumXY_loss)
		train_stepR = tf.train.AdamOptimizer(1e-5).minimize(rc_loss)
		train_step = tf.train.AdamOptimizer(1e-5).minimize(joint_loss)

	sess.run( tf.global_variables_initializer() )

	with tf.name_scope('accuracy'):
		correct_predictionX = tf.equal(tf.argmax(xc_conv, 1), tf.argmax(xc, 1))
		correct_predictionX = tf.cast(correct_predictionX, tf.float32)
		correct_predictionY = tf.equal(tf.argmax(yc_conv, 1), tf.argmax(yc, 1))
		correct_predictionY = tf.cast(correct_predictionY, tf.float32)
		correct_predictionR = tf.equal(tf.argmax(rc_conv, 1), tf.argmax(rc, 1))
		correct_predictionR = tf.cast(correct_predictionR, tf.float32)
		#accuracy = tf.reduce_mean(correct_predictionX) \
		#	+ tf.reduce_mean(correct_predictionY) \
		#	+ tf.reduce_mean(correct_predictionR)
		accuracy = tf.reduce_mean(correct_predictionX)/3.0 \
			+ tf.reduce_mean(correct_predictionY)/3.0 \
			+ tf.reduce_mean(correct_predictionR)/3.0

	tf.add_to_collection( 'img', img )
	tf.add_to_collection( 'xc', xc )
	tf.add_to_collection( 'yc', yc )
	tf.add_to_collection( 'rc', rc )
	tf.add_to_collection( 'keep_prob', keep_prob )
	tf.add_to_collection( 'xc_conv', xc_conv )
	tf.add_to_collection( 'yc_conv', yc_conv )
	tf.add_to_collection( 'rc_conv', rc_conv )
	tf.add_to_collection( 'train_step', train_step )
	tf.add_to_collection( 'train_stepXY', train_stepXY )
	tf.add_to_collection( 'train_stepR', train_stepR )
	tf.add_to_collection( 'accuracy', accuracy )

	#original: return y_conv, keep_prob
	# return x_conv, y_conv, r_conv, keep_prob
	return


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
