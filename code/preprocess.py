import pickle
import numpy as np
import tensorflow as tf
import os
from PIL import Image

# May need more functions for preprocessing 

def unpickle(file):
	"""
	CIFAR data contains the files data_batch_1, data_batch_2, ..., 
	as well as test_batch. We have combined all train batches into one
	batch for you. Each of these files is a Python "pickled" 
	object produced with cPickle. The code below will open up a 
	"pickled" object (each file) and return a dictionary.

	:param file: the file to unpickle
	:return: dictionary of unpickled data
	"""
	images = []
	labels = []

	
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict


def smart_resize(input_image, new_size):
	"""
	Resizes and crops given image to a square with a width of new_size
	"""
	width = input_image.width
	height = input_image.height

	# Image is portrait or square
	if height >= width:
		crop_box = (0, (height-width)//2, width, (height-width)//2 + width)
		return input_image.resize(size = (new_size,new_size),box = crop_box)

	# Image is landscape
	if width > height:
		crop_box = ((width-height)//2, 0, (width-height)//2 + height, height)
		return input_image.resize(size = (new_size,new_size),box = crop_box)

def get_labels_from_folder_names():
	"""
	Extracts all the label names from the image folder names 
	"""
	root='./../data/'
	labels = [ item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]
	return labels


def get_data(file_path):
	"""
	Given a file path and two target classes, returns an array of 
	normalized inputs (images) and an array of labels. 
	You will want to first extract only the data that matches the 
	corresponding classes we want (there are 10 classes and we only want 2).
	You should make sure to normalize all inputs and also turn the labels
	into one hot vectors using tf.one_hot().
	# Note that because you are using tf.one_hot() for your labels, your
	# labels will be a Tensor, while your inputs will be a NumPy array. This 
	# is fine because TensorFlow works with NumPy arrays.
	:param file_path: file path for inputs and labels, something 
	# like 'CIFAR_data_compressed/train'
	# :param first_class:  an integer (0-9) representing the first target
	# class in the CIFAR10 dataset, for a cat, this would be a 3
	:param first_class:  an integer (0-9) representing the second target
	class in the CIFAR10 dataset, for a dog, this would be a 5
	:return: normalized NumPy array of inputs and tensor of labels, where 
	inputs are of type np.float32 and has size (num_inputs, width, height, num_channels) and labels 
	has size (num_examples, num_cl
	num_classes = 0sses)
	"""
	image_size = 32
	unpickled_file = unpickle(file_path)
	inputs = unpickled_file[b'data']
	inputs = [smart_resize(input_image, image_size) for input_image in inputs]
	# labels = unpickled_file[b'labels']
	labels = get_labels_from_folder_names()
	num_classes = len(labels)

	# Getting all labels that are for food images
	labels = np.array(labels)

	# Reshape and transpose inputs
	# inputs = tf.reshape(inputs, (-1, 3)) #, 32 ,32))
	# inputs = tf.transpose(inputs, perm=[0,2,3,1])
	inputs = np.float32(inputs/255)

	# One-hot encoding for labels 
	labels = tf.one_hot(labels, 2)

	return inputs, labels