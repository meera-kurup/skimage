import numpy as np
import tensorflow as tf
from PIL import Image

def get_data(img_file, labels_file, image_size):
	"""
	Given a file path and two target classes, returns an array of 
	normalized inputs (images) and an array of labels. 
	Extract only the data that matches the corresponding classes
	(there are 101 classes and we only want 5).
	Normalizes all inputs and also turns the labels
	into one hot vectors using tf.one_hot().
	
	:param img_file: file path for inputs
	:param labels_file: file path for labels
	:param image_size: size of each input image
	"""
	print("Loading data")
	inputs = np.load(img_file, allow_pickle=True)
	labels = np.load(labels_file, allow_pickle=True)

	img = Image.fromarray(inputs[0], 'RGB')
	#img.show()
	# inputs = inputs/255
	# test_inputs = np.float32(test_inputs/255)

	# One-hot encoding for labels 
	d_str = np.unique(labels)
	# label_dict = dict(enumerate(d_str.flatten(), 0))
	label_dict = dict(zip(d_str.flatten(), range(len(d_str))))

	# num_classes = len(label_dict)

	# Only process 5 classes
	labels = np.vectorize(label_dict.get)(labels)
	processed_labels = labels[((labels == 0) | (labels == 1)) | (labels == 2) | (labels == 3) | (labels == 4)]
	temp_labels = np.where(processed_labels == 1, 1, 0)
	temp_labels2 = np.where(processed_labels == 2, 2, 0)
	temp_labels3 = np.where(processed_labels == 3, 3, 0)
	temp_labels4 = np.where(processed_labels == 4, 4, 0)
	processed_labels = temp_labels + temp_labels2 + temp_labels3 + temp_labels4
	one_hot = tf.one_hot(processed_labels, depth=5)

	processed_inputs = inputs[((labels == 0) | (labels == 1)) | (labels == 2) | (labels == 3) | (labels == 4)]
	processed_inputs = np.array(processed_inputs/255)
	processed_inputs = tf.reshape(processed_inputs, (-1, 3, image_size, image_size))
	processed_inputs = tf.transpose(processed_inputs, perm=[0,2,3,1])
	#print(processed_inputs.dtype)
	processed_inputs = tf.dtypes.cast(processed_inputs, tf.float32)

	# labels = tf.one_hot(np.vectorize(label_dict.get)(labels), num_classes)
	# test_labels = tf.one_hot(np.vectorize(label_dict.get)(test_labels), num_classes)

	return processed_inputs, one_hot