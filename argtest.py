import argparse as ap
# Importing library that supports user friendly commandline interfaces
import cv2
# Importing the opencv library
import imutils
# Importing the library that supports basic image processing functions
import numpy as np
# Importing the array operations library for python
import os
# Importing the library which supports standard systems commands
from scipy.cluster.vq import *
# Importing the library which classifies set of observations into clusters
from sklearn.preprocessing import StandardScaler
# Importing the library that supports centering and scaling vectors
from imutils import paths
import sys
sys.path.append('/home/pra/.virtualenvs/cv/lib/python2.7/site-packages/')

im=cv2.imread('/home/pra/2sem/mp/a2/dataset/training/horses/horse1.jpg')
print(im.shape)
