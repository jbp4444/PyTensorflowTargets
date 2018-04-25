
import numpy as np
import argparse

import targetlib

if( __name__ == '__main__' ):

	parser = argparse.ArgumentParser( description='display a simulated target image' )
	parser.add_argument( '-z', action='store_const', const=1,
			help='initialize image to zeroes, not random' )
	parser.add_argument( '-T', nargs='?', type=int, default=1,
			help='target type (1=O shape, 2=P shape)' )
	parser.add_argument( '-R', nargs='?', type=int, default=0,
			help='target rotation (0 to 3)' )
	parser.add_argument( '-x', nargs='?', type=int, default=14,
			help='x-coord for target' )
	parser.add_argument( '-y', nargs='?', type=int, default=14,
			help='y-coord for target' )
	parser.add_argument( '-n', nargs='?', type=float, default=0.1,
			help='noise-level for val=0 (0.0 up to x)' )
	parser.add_argument( '-N', nargs='?', type=float, default=0.9,
			help='noise-level for val=1 (x up to 1.0)' )
	parser.add_argument( '-v', action='count', help='verbose output' )
	parser.add_argument( '-V', action='count', help='really verbose output' )

	params = vars( parser.parse_args() )
	#print( params )

	targetlib.globals.TARGET_TYPE = params['T']
	verbose = 0
	if( params['v'] != None ):
		verbose += params['v']
	if( params['V'] != None ):
		verbose += 10*params['V']

	if( params['z'] != None ):
		img = np.zeros( (1, 28*28), np.float32 )
	else:
		img = params['n'] * np.random.random( (1,28*28) )

	targetlib.gen_data.draw_one_image( img, 0, params['x'],params['y'], params['R'], params['N'] )

	if( verbose ):
		i = 0
		for r in range(28):
			txt = ''
			for c in range(28):
				val = " %4.2f" % img[0][i]
				txt = txt + val
				i = i + 1
			print txt

		print
		print

	i = 0
	for r in range(28):
		txt = ''
		for c in range(28):
			if( img[0][i] > 0.5 ):
				txt = txt + 'X'
			else:
				txt = txt + '.'
			i = i + 1
		print txt
