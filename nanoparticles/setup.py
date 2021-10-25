################################################################################
# SETUP CYTHON
################################################################################

import numpy
from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules  = cythonize("nanoparticle_cython.pyx", annotate = True),
      include_dirs = [numpy.get_include()])

################################################################################
# END
################################################################################
