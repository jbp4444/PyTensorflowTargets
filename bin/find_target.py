

import argparse
import tensorflow as tf

import targetlib

if( __name__ == '__main__' ):
	parser = argparse.ArgumentParser( description='use TensorFlow to find a target in a simulated image' )
	parser.add_argument( '-k', nargs='?', type=float, default=0.5,
			help='keep-probability for training' )
	parser.add_argument( '-e', nargs='?', type=int, default=1000,
			help='number of epochs to run' )
	parser.add_argument( '-b', nargs='?', type=int, default=100,
			help='batch size per epoch' )
	parser.add_argument( '-T', nargs='?', type=int, default=1,
			help='target type (1=O shape, 2=P shape)' )
	parser.add_argument( '-n', nargs='?', type=float, default=0.1,
			help='noise-level for val=0 (0.0 up to x)' )
	parser.add_argument( '-N', nargs='?', type=float, default=0.9,
			help='noise-level for val=1 (x up to 1.0)' )
	parser.add_argument( '-v', action='count', help='verbose output' )
	parser.add_argument( '-V', action='count', help='really verbose output' )

	params = vars( parser.parse_args() )
	#print( params )

	targetlib.globals.KEEP_PROB   = params['k']
	targetlib.globals.NUM_EPOCHS  = params['e']
	targetlib.globals.MYDATA_BATCH_SIZE = params['b']
	targetlib.globals.NOISE_LVL0  = params['n']
	targetlib.globals.NOISE_LVL1  = params['N']
	targetlib.globals.TARGET_TYPE = params['T']

	#targetlib.globals.VERBOSE = 0
	if( params['v'] != None ):
		targetlib.globals.VERBOSE += params['v']
	if( params['V'] != None ):
		targetlib.globals.VERBOSE += 10*params['V']

	tf.app.run( main=targetlib.main.main )
