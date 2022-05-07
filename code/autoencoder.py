import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D, BatchNormalization, AveragePooling2D, Input
from keras.applications.inception_v3 import InceptionV3
from keras.regularizers import l2