#!/usr/bin/python


import sys
import exceptions
from optparse import *
import glob
import re

import numpy
import cPickle

from matplotlib import pyplot

##############################################################################
# parse options and arguments
usage = """Usage: %prog [options] 

Use bioreactor simulation model to generate evolution plots
"""
parser = OptionParser(usage=usage, version="")
parser.add_option("-g", "--glob-found", action="store", type="string", \
                  default=None, metavar=" FOUND_GLOB", \
                  help="GLOB of found trigger/injection files to read")


(opts, args) = parser.parse_args()