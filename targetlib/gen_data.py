# Copyright (C) 2018, John Pormann, Duke University Libraries
#
# The "images" are 28x28 grids of floats, 0.0 to 1.0 (i.e. black and white images, not color)
# Noise is added to the overall image using random.uniform()

# for target-type == 1:
#
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


# for target-type==2, the target is a P-shape:
# ............................
# ..###.......................
# ..#.#.......................
# ..###.......................
# ..#.........................
# ..#.........................
# ............................
#
# the hope is that the longer "tail" will be easier for TF to identify
#
# the pxlist, alist, blist refer to the following; for rotation==0:
# ............................
# ..PPP.......................
# ..P.P.......................
# ..PPP.......................
# ..A.........................
# ..B.........................
# ............................
# for rotation==1:
# .............................
# .BAPPP.......................
# ...P.P.......................
# ...PPP.......................
# .............................
# for rotation==2:
# ............................
# ....B.......................
# ....A.......................
# ..PPP.......................
# ..P.P.......................
# ..PPP.......................
# ............................
# for rotation==3:
# ............................
# ..PPP.......................
# ..P.P.......................
# ..PPPAB.....................
# ............................


import random

import numpy as np
import tensorflow as tf

import targetlib

# fixed/true target locations for test set
# : pixel list for the box(es)
pxlist = [ [-1,-1], [-1,0], [-1,1], [0,-1], [0,1], [1,-1], [1,0], [1,1] ]
# : pixel lists for the 'tabs'
alist1  = [ [-1,2], [-2,-1], [1,-2], [2,1] ]
blist1  = [ [2,-1],  [1,2],   [-2,1], [-1,-2] ]
# pixel lists for the box and 'tabs'
alist2  = [ [-1,2], [-2,-1], [1,-2], [2,1] ]
blist2  = [ [-1,3], [-3,-1], [1,-3], [3,1] ]
# TODO: yes, these two should be inside an array

# with target-type==1, we could go closer to edges
test_xylist = [ [3,3], [3,24], [24,3], [24,24], [13,23], [21,17] ]
test_xlist = []
test_ylist = []
test_rlist = []

# min,max locations for the targets, by target-type
search_space = [
	(),  # type=0 is not defined
	(2,24),
	(3,24),
	( targetlib.globals.TYPE3_SEP+1, 28-targetlib.globals.TYPE3_SEP-2 )
]

# pixel lists for the box and 'tabs'

# draws one image in-place in the set of imgs
# : while it would be more readable to embed the noise-lvl0 stuff in here,
#   it should be faster to build the large zero/random tensor in one shot
#   (i.e. outside of this function)
def draw_one_image_type3( img, i, xval,yval, rot, noise_lvl1 ):

	# draw the boxes
	for ix,iy in pxlist:
		img[ i, (yval+iy)*28 + xval+ix ] = random.uniform(noise_lvl1,1)

	# TODO: check xval,yval to ensure that 2nd/3rd targets don't fall out-of-bounds
	sep = targetlib.globals.TYPE3_SEP

	if rot == 0:
		xv = xval + sep
		yv = yval
		for ix,iy in pxlist:
			img[ i, (yv+iy)*28 + xv+ix ] = random.uniform(noise_lvl1,1)
		xv = xval
		yv = yval + sep
		for ix,iy in pxlist:
			img[ i, (yv+iy)*28 + xv+ix ] = random.uniform(noise_lvl1,1)

	elif rot == 1:
		xv = xval
		yv = yval + sep
		for ix,iy in pxlist:
			img[ i, (yv+iy)*28 + xv+ix ] = random.uniform(noise_lvl1,1)
		xv = xval - sep
		yv = yval
		for ix,iy in pxlist:
			img[ i, (yv+iy)*28 + xv+ix ] = random.uniform(noise_lvl1,1)

	elif rot == 2:
		xv = xval - sep
		yv = yval
		for ix,iy in pxlist:
			img[ i, (yv+iy)*28 + xv+ix ] = random.uniform(noise_lvl1,1)
		xv = xval
		yv = yval - sep
		for ix,iy in pxlist:
			img[ i, (yv+iy)*28 + xv+ix ] = random.uniform(noise_lvl1,1)

	elif rot == 3:
		xv = xval
		yv = yval - 10
		for ix,iy in pxlist:
			img[ i, (yv+iy)*28 + xv+ix ] = random.uniform(noise_lvl1,1)
		xv = xval + 10
		yv = yval
		for ix,iy in pxlist:
			img[ i, (yv+iy)*28 + xv+ix ] = random.uniform(noise_lvl1,1)

def draw_one_image( img, i, xval,yval, rot, noise_lvl1 ):
	if targetlib.globals.TARGET_TYPE == 1:
		ax,ay = alist1[rot]
		bx,by = blist1[rot]
	elif targetlib.globals.TARGET_TYPE == 2:
		ax,ay = alist2[rot]
		bx,by = blist2[rot]
	elif targetlib.globals.TARGET_TYPE == 3:
		draw_one_image_type3( img, i, xval,yval, rot, noise_lvl1 )
		return

	# draw the box
	for ix,iy in pxlist:
		img[ i, (yval+iy)*28 + xval+ix ] = random.uniform(noise_lvl1,1)

	# add the 'wings'
	img[ i, (yval+ay)*28 + xval+ax ] = random.uniform(noise_lvl1,1)
	img[ i, (yval+by)*28 + xval+bx ] = random.uniform(noise_lvl1,1)

def my_test_data():
	print( 'my_test_data called' )
	noise_lvl0 = targetlib.globals.NOISE_LVL0
	noise_lvl1 = targetlib.globals.NOISE_LVL1
	batch_sz = len(test_xylist) * 4

	img = noise_lvl0*np.random.random( (batch_sz,28*28) )
	# for clean/no-noise data:  img = np.zeros( (batch_sz, 28*28), np.float32 )
	xout = np.zeros( (batch_sz,28), np.float32 )
	yout = np.zeros( (batch_sz,28), np.float32 )
	rout = np.zeros( (batch_sz,4), np.float32 )

	mn,mx = search_space[targetlib.globals.TARGET_TYPE]
	#print( 'min,max for search space = ',mn,mx )

	idx = 0
	for i in range(len(test_xylist)):
		xval,yval = test_xylist[i]
		#print( 'xval,yval = ',xval,yval )

		if xval < mn:
			xval = mn
		if xval > mx:
			xval = mx
		if yval < mn:
			yval = mn
		if yval > mx:
			yval = mx
		#print( '  new xval,yval = ',xval,yval )


		for j in range(4):
			draw_one_image( img, idx, xval,yval, j, 1.0 )

			test_xlist.append( xval )
			test_ylist.append( yval )
			test_rlist.append( j )

			# this should produce one-hot outputs
			xout[idx,xval] = 1
			yout[idx,yval] = 1
			rout[idx,j]    = 1

			idx = idx + 1

	return ( img, xout,yout, rout )

def my_test_data_noise():
	print( 'my_test_data_noise called' )
	noise_lvl0 = targetlib.globals.NOISE_LVL0
	noise_lvl1 = targetlib.globals.NOISE_LVL1
	batch_sz = len(test_xylist) * 4

	img = np.random.random( (batch_sz,28*28) )
	xout = np.random.random( (batch_sz,28) )
	yout = np.random.random( (batch_sz,28) )
	rout = np.random.random( (batch_sz,4) )

	return ( img, xout,yout, rout )

def my_data_gen():
	print( 'my_data_gen called' )
	counter = 0
	batch_sz   = targetlib.globals.MYDATA_BATCH_SIZE
	noise_lvl0 = targetlib.globals.NOISE_LVL0
	noise_lvl1 = targetlib.globals.NOISE_LVL1

	mn,mx = search_space[targetlib.globals.TARGET_TYPE]
	#print( 'min,max for search space = ',mn,mx )

	while True:
		#img = [ 0 for i in range(28*28) ]
		#img = np.zeros( (batch_sz, 28*28), np.float32 )
		img = noise_lvl0*np.random.random( (batch_sz,28*28) )
		xout = np.zeros( (batch_sz,28), np.float32 )
		yout = np.zeros( (batch_sz,28), np.float32 )
		rout = np.zeros( (batch_sz,4), np.float32 )

		for i in range(batch_sz):
			xval = random.randint(mn,mx)    # avoid edges for now
			yval = random.randint(mn,mx)    # avoid edges for now
			rot  = random.randint(0,3)     # 0123==NSEW rotation

			draw_one_image( img, i, xval,yval, rot, noise_lvl1 )

			# this should produce one-hot outputs
			xout[i,xval] = 1
			yout[i,yval] = 1
			rout[i,rot]  = 1

		#if( (counter%100)==0 ):
		#	print( 'my_data_gen ctr=',counter )
		counter = counter + 1

		yield ( img, xout,yout, rout )
