# Python/TensorFlow - Target Detection
----

This repository contains Python/TensorFlow code for doing target tracking.
The image-field is 28x28 (following the TF MNIST example), and targets are
generated inside a generator-function ... so you can easily train on 100's
or 1000's of targets without having to pre-generate them.

* bin/find_target.py - the main driver program
* bin/print_image.py - shows a text-pixel image of a target

* targetlib - the main set of TF code
  * gen_data.py - the generator for the target data
  * globals.py - keeps global variables like noise levels, target-type, and some TF metavariables
  * nnet.py - builds the CNN to track (x,y) location of the target as well as the rotation (NSEW)

* automatically stores and restores the model from disk
