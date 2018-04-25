from setuptools import setup

setup(
   name='targetlib',
   version='0.1',
   description='A TensorFlow module to find a target in an image',
   author='JBP',
   author_email='jbp@example.com',
   packages=['targetlib'],  #same as name
   scripts=[ 'bin/find_target.py' ],  # executables
   install_requires=[ 'tensorflow' ], #external packages as dependencies
   test_suite = 'nose.collector',
   tests_require = [ 'nose' ],
)
