# Python/TensorFlow - Target Detection
----

This repository contains Python/TensorFlow code for doing target tracking.
The image-field is 28x28 (following the TF MNIST example), and targets are
generated inside a generator-function ... so you can easily train on 100's
or 1000's of targets without having to pre-generate them.

* targetlib - the main set of TF code
  * gen_data.py - the generator for the target data; also provides a set of test data (a set of (x,y) coords times 4, one for each rotation, NSEW)
    * images are just 28x28 grids of 0.0 to 1.0
	* a zero is assigned a uniform random value, 0.0 to NOISE_LVL0
	* a one is assigned a uniform random value, NOISE_LVL1 to 1.0
    * TARGET_TYPE==1 is a 3x3 box shape with a couple of "wings" to give it an orientation
    * TARGET_TYPE==2 is a P shape (3x3 top with a 2-pixel tail)
	* TARGET_TYPE==3 is a set of three 3x3 boxes in a square
  * globals.py - keeps global variables like noise levels, target-type, and some TF metavariables
  * nnet.py - builds the CNN to track (x,y) location of the target as well as the rotation (NSEW)

* bin/find_target.py - the main driver program
  * automatically stores and restores the model from disk
* bin/print_image.py - shows a text-pixel image of a target
  * see TARGET_TYPE's above

* results on locating the coordinates of the box are very good (99%++); identifying the rotation takes longer to train, but can also achieve very good accuracy too (85%+)
