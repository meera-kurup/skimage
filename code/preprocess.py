import numpy as np
import tensorflow as tf
from PIL import Image
# May need more functions for preprocessing 

# def smart_resize(input_image, new_size):
# 	"""
# 	Resizes and crops given image to a square with a width of new_size
# 	"""
# 	width = input_image.width
# 	height = input_image.height

# 	# Image is portrait or square
# 	if height >= width:
# 		crop_box = (0, (height-width)//2, width, (height-width)//2 + width)
# 		return input_image.resize(size = (new_size,new_size),box = crop_box)

# 	# Image is landscape
# 	if width > height:
# 		crop_box = ((width-height)//2, 0, (width-height)//2 + height, height)
# 		return input_image.resize(size = (new_size,new_size),box = crop_box)

# def get_labels_from_folder_names():
# 	"""
# 	Extracts all the label names from the image folder names 
# 	"""
# 	root='./../data/'
# 	labels = [ item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]
# 	return labels

def get_data(img_file, labels_file):
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
	image_size = 256
	# unpickled_file = unpickle(file_path)
	# inputs = unpickled_file[b'data']
	# inputs = [smart_resize(input_image, image_size) for input_image in inputs]
	# # labels = unpickled_file[b'labels']
	# labels = get_labels_from_folder_names()
	print("Loading data...")
	inputs = np.load(img_file, allow_pickle=True)
	labels = np.load(labels_file, allow_pickle=True)
	# print("Loading testing data...")
	# test_inputs = np.load(test_img_file)
	# test_labels = np.load(test_labels_file)
	img = Image.fromarray(inputs[0], 'RGB')
	img.show()
	# Getting all labels that are for food images
	# labels = np.array(labels)

	# Reshape and transpose inputs
	# inputs = tf.reshape(inputs, (-1, 3)) #, 32 ,32))
	# inputs = tf.transpose(inputs, perm=[0,2,3,1])
	img = Image.fromarray(inputs[0], 'RGB')
	img.show()
	inputs = inputs/255
	# test_inputs = np.float32(test_inputs/255)
	# One-hot encoding for labels 
	d_str = np.unique(labels)
	# label_dict = dict(enumerate(d_str.flatten(), 0))
	label_dict = dict(zip(d_str.flatten(), range(len(d_str))))

	# num_classes = len(label_dict)

	labels = np.vectorize(label_dict.get)(labels)
	processed_labels = labels[((labels == 0) | (labels == 1)) | (labels == 2) | (labels == 3) | (labels == 4)]
	temp_labels = np.where(processed_labels == 1, 1, 0)
	temp_labels2 = np.where(processed_labels == 2, 2, 0)
	temp_labels3 = np.where(processed_labels == 3, 3, 0)
	temp_labels4 = np.where(processed_labels == 4, 4, 0)
	processed_labels = sum(temp_labels, temp_labels2, temp_labels3, temp_labels4)
	one_hot = tf.one_hot(processed_labels, depth=3)

	processed_inputs = inputs[((labels == 0) | (labels == 1)) | (labels == 2) | (labels == 3) | (labels == 4)]
	processed_inputs = np.array(processed_inputs/255)
	processed_inputs = tf.reshape(processed_inputs, (-1, 3, image_size, image_size))
	processed_inputs = tf.transpose(processed_inputs, perm=[0,2,3,1])
	processed_inputs = tf.dtypes.cast(processed_inputs, tf.float32)

	# first_class = 11
	# second_class = 21
	# train_labels = np.vectorize(label_dict.get)(train_labels)
	# indicies_1 = np.nonzero(train_labels == first_class)
	# indicies_2 = np.nonzero(train_labels == second_class)
	# indicies = np.concatenate([indicies_1[0], indicies_2[0]])
	# train_inputs = np.float32(train_inputs[indicies]/255)
	# train_labels = train_labels[indicies]
	# train_inputs = tf.reshape(train_inputs, (-1, 3, 299, 299))
	# train_inputs = tf.transpose(train_inputs, perm=[0,2,3,1])
	# train_labels = np.where(train_labels == first_class, 0, 1)

	# labels = tf.one_hot(labels, num_classes)

	# labels = tf.one_hot(np.vectorize(label_dict.get)(labels), num_classes)
	# test_labels = tf.one_hot(np.vectorize(label_dict.get)(test_labels), num_classes)

	return processed_inputs, one_hot