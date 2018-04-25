
#
# The "images" are 28x28 grids of floats, 0.0 to 1.0 (i.e. black and white images, not color)
# Noise is added to the overall image using random.uniform()

# a simple image might look like this (top 6 lines with target at (3,3)
# 0.04 0.02 0.01 0.09 0.01 0.08 0.01 0.03 0.05 0.05 0.03 0.10 0.05 0.06 0.02 ...
# 0.04 0.10 0.05 0.07 0.95 0.02 0.08 0.01 0.04 0.02 0.08 0.02 0.07 0.00 0.06 ...
# 0.08 0.07 0.96 0.94 0.97 0.06 0.04 0.07 0.03 0.07 0.08 0.04 0.04 0.06 0.06 ...
# 0.07 0.05 0.91 0.00 0.98 0.07 0.03 0.04 0.05 0.08 0.03 0.10 0.04 0.02 0.03 ...
# 0.04 0.94 1.00 0.91 0.95 0.05 0.02 0.06 0.02 0.01 0.03 0.02 0.09 0.10 0.07 ...
# 0.08 0.09 0.04 0.09 0.00 0.05 0.05 0.10 0.05 0.03 0.04 0.08 0.01 0.05 0.07 ...
#
# or, more pixel-like:
# ............................
# ....#.......................
# ..###.......................
# ..#.#.......................
# .####.......................
# ............................
#
# the pxlist, alist, blist refer to the following; for rotation==0:
# ............................
# ..PPPB......................
# ..P.P.......................
# ..PPP.......................
# ..A.........................
# ............................
# for rotation==1:
# ............................
# .APPP.......................
# ..P.P.......................
# ..PPP.......................
# ....B.......................
# ............................
# for rotation==2:
# ............................
# ....A.......................
# ..PPP.......................
# ..P.P.......................
# .BPPP.......................
# ............................
# for rotation==3:
# ............................
# ..B.........................
# ..PPP.......................
# ..P.P.......................
# ..PPPA......................
# ............................


import random

import numpy as np
import tensorflow as tf

import targetlib

# fixed/true target locations for test set
test_xlist = []
test_ylist = []
test_rlist = []


# draws one image in-place in the set of imgs
def draw_one_image( img, i, xval,yval, rot, noise_lvl1 ):
	# extra tabs on the target ("A" and "B")
	ax,ay = alist[rot]
	bx,by = blist[rot]

	# draw the box
	for ix,iy in pxlist:
		img[ i, (yval+iy)*28 + xval+ix ] = random.uniform(noise_lvl1,1)

	# add the 'wings'
	img[ i, (yval+ay)*28 + xval+ax ] = random.uniform(noise_lvl1,1)
	img[ i, (yval+by)*28 + xval+bx ] = random.uniform(noise_lvl1,1)


def my_test_data():
	print( 'my_test_data called' )
	batch_sz = len(test_xylist) * len(alist)

	#img = [ 0 for i in range(28*28) ]
	img = np.zeros( (batch_sz, 28*28), np.float32 )
	xout = np.zeros( (batch_sz,28), np.float32 )
	yout = np.zeros( (batch_sz,28), np.float32 )
	rout = np.zeros( (batch_sz,4), np.float32 )

	idx = 0
	for i in range(len(test_xylist)):
		xval,yval = test_xylist[i]

		for j in range(len(alist)):
			draw_one_image( img, idx, xval,yval, j, 1.0 )

			test_xlist.append( xval )
			test_ylist.append( yval )
			test_rlist.append( j )

			# this should produce one-hot outputs
			xout[idx,xval] = 1
			yout[idx,yval] = 1
			rout[idx,j]   = 1

			idx = idx + 1

	return ( img, xout,yout, rout )

def my_data_gen():
	print( 'my_data_gen called' )
	counter = 0
	batch_sz = targetlib.globals.MYDATA_BATCH_SIZE
	noise_lvl0 = targetlib.globals.NOISE_LVL0
	noise_lvl1 = targetlib.globals.NOISE_LVL1

	while True:
		#img = [ 0 for i in range(28*28) ]
		#img = np.zeros( (batch_sz, 28*28), np.float32 )
		img = noise_lvl0*np.random.random( (batch_sz,28*28) )
		xout = np.zeros( (batch_sz,28), np.float32 )
		yout = np.zeros( (batch_sz,28), np.float32 )
		rout = np.zeros( (batch_sz,4), np.float32 )

		for i in range(batch_sz):
			xval = random.randint(2,25)    # avoid edges for now
			yval = random.randint(2,25)    # avoid edges for now
			rot  = random.randint(0,3)     # 0123==NSEW rotation

			draw_one_image( img, i, xval,yval, rot, noise_lvl1 )

			# this should produce one-hot outputs
			xout[i,xval] = 1
			yout[i,yval] = 1
			rout[i,rot]  = 1

		if( (counter%100)==0 ):
			print( 'my_data_gen ctr=',counter )
		counter = counter + 1

		yield ( img, xout,yout, rout )
