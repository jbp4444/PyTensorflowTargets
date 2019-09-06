# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
        also includes rotation prediction (1 of 4, NSEW)

"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function


import numpy as np
import tensorflow as tf

import targetlib


def main( args ):

	try:
		new_saver = tf.train.import_meta_graph( './my-model.meta' )
		print( "reading existing nnet" )
	except:
		print( "creating nnet from scratch" )
		with tf.Session() as sess:
			# Create the model
			targetlib.nnet.create_deepnn( sess )

			print( 'saving model to disk' )
			saver0 = tf.train.Saver()
			saver0.save( sess, './my-model' )
			saver0.export_meta_graph( './my-model.meta' )


	# create the test data once and save it
	test_images,test_xvals,test_yvals,test_rvals = targetlib.gen_data.my_test_data()
	# : and some non-target images for testing
	test_images2,test_xvals2,test_yvals2,test_rvals2 = targetlib.gen_data.my_test_data_noise()

	# start the noise levels off low and then ramp them up?
	if( targetlib.globals.NOISE_LVL_RAMP < 0 ):
		targetlib.globals.NOISE_LVL0 = targetlib.globals.INIT_NOISE_LVL0
		targetlib.globals.NOISE_LVL1 = targetlib.globals.INIT_NOISE_LVL1
	else:
		targetlib.globals.NOISE_LVL0 = 0.0
		targetlib.globals.NOISE_LVL1 = 0.0

	# iterator/in-line data-generator
	my_iter = targetlib.gen_data.my_data_gen()

	with tf.Session() as sess:

		try:
			new_saver = tf.train.Saver()
			new_saver.restore( sess, './my-model' )

  			test_writer = tf.summary.FileWriter( './logs')

		except Exception as e:
			print( "ERROR - "+str(e) )

		graph = tf.get_default_graph()

		#opers = graph.get_operations()
		#for o in opers:
		#	print( o )

		img = graph.get_tensor_by_name( 'img:0' )
		xc  = graph.get_tensor_by_name( 'xc:0' )
		yc  = graph.get_tensor_by_name( 'yc:0' )
		rc  = graph.get_tensor_by_name( 'rc:0' )
		keep_prob  = graph.get_tensor_by_name( 'keep_prob:0' )

		xc_conv = tf.get_collection( 'xc_conv' )[0]
		yc_conv = tf.get_collection( 'yc_conv' )[0]
		rc_conv = tf.get_collection( 'rc_conv' )[0]
		#x_image_node = graph.get_tensor_by_name( 'reshape/x_image:0' )
		#loss_node = graph.get_tensor_by_name( 'loss/joint_loss:0' )
		train_step = tf.get_collection( 'train_step' )[0]

		accuracy = tf.get_collection( 'accuracy' )[0]

		for i in range( targetlib.globals.NUM_EPOCHS ):
			# ramp-up noise-levels?
			if( i <= targetlib.globals.NOISE_LVL_RAMP ):
				pct = (i+0.0) / targetlib.globals.NOISE_LVL_RAMP
				if( pct > 1.0 ):
					pct = 1.0
				targetlib.globals.NOISE_LVL0 = pct*targetlib.globals.INIT_NOISE_LVL0
				targetlib.globals.NOISE_LVL1 = pct*targetlib.globals.INIT_NOISE_LVL1
				print( "noise-levels mult set to %.2f" % pct )

			#batch = mnist.train.next_batch(50)
			(img_vals,xc_vals,yc_vals,rc_vals) = my_iter.next()

			if i % 100 == 0:
				train_accuracy = accuracy.eval( feed_dict={
					img: img_vals,   #batch[0],
					xc:  xc_vals,    #batch[1],
					yc:  yc_vals,
					rc:  rc_vals,
					keep_prob: 1.0
				})
				print('step %d, training accuracy %g' % (i, train_accuracy))

			train_step.run( feed_dict={
				img: img_vals,   #batch[0],
				xc:  xc_vals,    #batch[1],
				yc:  yc_vals,
				rc:  rc_vals,
				keep_prob: targetlib.globals.KEEP_PROB
			})


		# print out final results on test data
		print('final accuracy on test-set %g' % accuracy.eval( feed_dict={
			img: test_images,
			xc:  test_xvals,
			yc:  test_yvals,
			rc:  test_rvals,
			keep_prob: 1.0
		}))
		xxx = xc_conv.eval(feed_dict={
			img: test_images,
			xc:  test_xvals,
			yc:  test_yvals,
			rc:  test_rvals,
			keep_prob: 1.0
		})
		yyy = yc_conv.eval(feed_dict={
			img: test_images,
			xc:  test_xvals,
			yc:  test_yvals,
			rc:  test_rvals,
			keep_prob: 1.0
		})
		rrr = rc_conv.eval(feed_dict={
			img: test_images,
			xc:  test_xvals,
			yc:  test_yvals,
			rc:  test_rvals,
			keep_prob: 1.0
		})
		print( 'X:', np.argmax(xxx,1).tolist() )
		print( '  ', targetlib.gen_data.test_xlist )
		print( 'Y:', np.argmax(yyy,1).tolist() )
		print( '  ', targetlib.gen_data.test_ylist )
		print( 'R:', np.argmax(rrr,1).tolist() )
		print( '  ', targetlib.gen_data.test_rlist )

		# test with the all-noise images
		print('final accuracy on noise-set %g' % accuracy.eval( feed_dict={
			img: test_images2,
			xc:  test_xvals2,
			yc:  test_yvals2,
			rc:  test_rvals2,
			keep_prob: 1.0
		}))
		xxx = xc_conv.eval(feed_dict={
			img: test_images2,
			xc:  test_xvals2,
			yc:  test_yvals2,
			rc:  test_rvals2,
			keep_prob: 1.0
		})
		yyy = yc_conv.eval(feed_dict={
			img: test_images2,
			xc:  test_xvals2,
			yc:  test_yvals2,
			rc:  test_rvals2,
			keep_prob: 1.0
		})
		rrr = rc_conv.eval(feed_dict={
			img: test_images2,
			xc:  test_xvals2,
			yc:  test_yvals2,
			rc:  test_rvals2,
			keep_prob: 1.0
		})
		print( 'X:', np.argmax(xxx,1).tolist() )
		print( 'Y:', np.argmax(yyy,1).tolist() )
		print( 'R:', np.argmax(rrr,1).tolist() )


		# now save the model to disk
		print( 'saving model to disk' )
		saver0 = tf.train.Saver()
		saver0.save( sess, './my-model' )
		saver0.export_meta_graph( './my-model.meta' )

def main2( args ):

	try:
		new_saver = tf.train.import_meta_graph( './my-model.meta' )
		print( "reading existing nnet" )
	except:
		print( "creating nnet from scratch" )
		with tf.Session() as sess:
			# Create the model
			targetlib.nnet.create_deepnn( sess )

			print( 'saving model to disk' )
			saver0 = tf.train.Saver()
			saver0.save( sess, './my-model' )
			saver0.export_meta_graph( './my-model.meta' )


	# create the test data once and save it
	test_images,test_xvals,test_yvals,test_rvals = targetlib.gen_data.my_test_data()
	# : and some non-target images for testing
	test_images2,test_xvals2,test_yvals2,test_rvals2 = targetlib.gen_data.my_test_data_noise()

	# start the noise levels off low and then ramp them up?
	if( targetlib.globals.NOISE_LVL_RAMP < 0 ):
		targetlib.globals.NOISE_LVL0 = targetlib.globals.INIT_NOISE_LVL0
		targetlib.globals.NOISE_LVL1 = targetlib.globals.INIT_NOISE_LVL1
	else:
		targetlib.globals.NOISE_LVL0 = 0.0
		targetlib.globals.NOISE_LVL1 = 0.0

	# iterator/in-line data-generator
	my_iter = targetlib.gen_data.my_data_gen()

	with tf.Session() as sess:

		try:
			new_saver = tf.train.Saver()
			new_saver.restore( sess, './my-model' )

  			test_writer = tf.summary.FileWriter( './logs')

		except Exception as e:
			print( "ERROR - "+str(e) )

		graph = tf.get_default_graph()

		#opers = graph.get_operations()
		#for o in opers:
		#	print( o )

		img = graph.get_tensor_by_name( 'img:0' )
		xc  = graph.get_tensor_by_name( 'xc:0' )
		yc  = graph.get_tensor_by_name( 'yc:0' )
		rc  = graph.get_tensor_by_name( 'rc:0' )
		keep_prob  = graph.get_tensor_by_name( 'keep_prob:0' )

		xc_conv = tf.get_collection( 'xc_conv' )[0]
		yc_conv = tf.get_collection( 'yc_conv' )[0]
		rc_conv = tf.get_collection( 'rc_conv' )[0]
		#x_image_node = graph.get_tensor_by_name( 'reshape/x_image:0' )
		#loss_node = graph.get_tensor_by_name( 'loss/joint_loss:0' )
		train_step = tf.get_collection( 'train_step' )[0]
		train_stepR = tf.get_collection( 'train_stepR' )[0]
		train_stepXY = tf.get_collection( 'train_stepXY' )[0]

		accuracy = tf.get_collection( 'accuracy' )[0]

		flipflop = 0
		for i in range( targetlib.globals.NUM_EPOCHS ):
			# ramp-up noise-levels?
			if( i <= targetlib.globals.NOISE_LVL_RAMP ):
				pct = (i+0.0) / targetlib.globals.NOISE_LVL_RAMP
				if( pct > 1.0 ):
					pct = 1.0
				targetlib.globals.NOISE_LVL0 = pct*targetlib.globals.INIT_NOISE_LVL0
				targetlib.globals.NOISE_LVL1 = pct*targetlib.globals.INIT_NOISE_LVL1
				print( "noise-levels mult set to %.2f" % pct )

			#batch = mnist.train.next_batch(50)
			(img_vals,xc_vals,yc_vals,rc_vals) = my_iter.next()

			if i % 100 == 0:
				train_accuracy = accuracy.eval( feed_dict={
					img: img_vals,   #batch[0],
					xc:  xc_vals,    #batch[1],
					yc:  yc_vals,
					rc:  rc_vals,
					keep_prob: 1.0
				})
				print('step %d, training accuracy %g' % (i, train_accuracy))

			if( flipflop == 0 ):
				train_stepXY.run( feed_dict={
					img: img_vals,   #batch[0],
					xc:  xc_vals,    #batch[1],
					yc:  yc_vals,
					rc:  rc_vals,
					keep_prob: targetlib.globals.KEEP_PROB
				})
				flipflop = 1
			else:
				train_stepR.run( feed_dict={
					img: img_vals,   #batch[0],
					xc:  xc_vals,    #batch[1],
					yc:  yc_vals,
					rc:  rc_vals,
					keep_prob: targetlib.globals.KEEP_PROB
				})
				flipflop = 0



		print('final accuracy on test-set %g' % accuracy.eval( feed_dict={
			img: test_images,
			xc:  test_xvals,
			yc:  test_yvals,
			rc:  test_rvals,
			keep_prob: 1.0
		}))

		# print out final results on test data
		xxx = xc_conv.eval(feed_dict={
			img: test_images,
			xc:  test_xvals,
			yc:  test_yvals,
			rc:  test_rvals,
			keep_prob: 1.0
		})
		yyy = yc_conv.eval(feed_dict={
			img: test_images,
			xc:  test_xvals,
			yc:  test_yvals,
			rc:  test_rvals,
			keep_prob: 1.0
		})
		rrr = rc_conv.eval(feed_dict={
			img: test_images,
			xc:  test_xvals,
			yc:  test_yvals,
			rc:  test_rvals,
			keep_prob: 1.0
		})
		print( 'X:', np.argmax(xxx,1).tolist() )
		print( '  ', targetlib.gen_data.test_xlist )
		print( 'Y:', np.argmax(yyy,1).tolist() )
		print( '  ', targetlib.gen_data.test_ylist )
		print( 'R:', np.argmax(rrr,1).tolist() )
		print( '  ', targetlib.gen_data.test_rlist )


		# test with the all-noise images
		print('final accuracy on noise-set %g' % accuracy.eval( feed_dict={
			img: test_images2,
			xc:  test_xvals2,
			yc:  test_yvals2,
			rc:  test_rvals2,
			keep_prob: 1.0
		}))
		xxx = xc_conv.eval(feed_dict={
			img: test_images2,
			xc:  test_xvals2,
			yc:  test_yvals2,
			rc:  test_rvals2,
			keep_prob: 1.0
		})
		yyy = yc_conv.eval(feed_dict={
			img: test_images2,
			xc:  test_xvals2,
			yc:  test_yvals2,
			rc:  test_rvals2,
			keep_prob: 1.0
		})
		rrr = rc_conv.eval(feed_dict={
			img: test_images2,
			xc:  test_xvals2,
			yc:  test_yvals2,
			rc:  test_rvals2,
			keep_prob: 1.0
		})
		print( 'X:', np.argmax(xxx,1).tolist() )
		print( 'Y:', np.argmax(yyy,1).tolist() )
		print( 'R:', np.argmax(rrr,1).tolist() )


		# now save the model to disk
		print( 'saving model to disk' )
		saver0 = tf.train.Saver()
		saver0.save( sess, './my-model' )
		saver0.export_meta_graph( './my-model.meta' )
