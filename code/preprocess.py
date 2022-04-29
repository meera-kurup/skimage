### FROM MACHINE TRANSLATION PROJECT ###
import numpy as np
import tensorflow as tf
import numpy as np

from attenvis import AttentionVis
av = AttentionVis()

##########DO NOT CHANGE#####################
PAD_TOKEN = "*PAD*"
STOP_TOKEN = "*STOP*"
START_TOKEN = "*START*"
UNK_TOKEN = "*UNK*"
FRENCH_WINDOW_SIZE = 14
ENGLISH_WINDOW_SIZE = 14
##########DO NOT CHANGE#####################

def pad_corpus(french, english):
	"""
	DO NOT CHANGE:
	arguments are lists of FRENCH, ENGLISH sentences. Returns [FRENCH-sents, ENGLISH-sents]. The
	text is given an initial "*STOP*". English is padded with "*START*" at the beginning for Teacher Forcing.
	:param french: list of French sentences
	:param french: list of French sentences
	:param english: list of English sentences
	:return: A tuple of: (list of padded sentences for French, list of padded sentences for English)
	"""
	FRENCH_padded_sentences = []
	for line in french:
		padded_FRENCH = line[:FRENCH_WINDOW_SIZE]
		padded_FRENCH += [STOP_TOKEN] + [PAD_TOKEN] * (FRENCH_WINDOW_SIZE - len(padded_FRENCH)-1)
		FRENCH_padded_sentences.append(padded_FRENCH)

	ENGLISH_padded_sentences = []
	for line in english:
		padded_ENGLISH = line[:ENGLISH_WINDOW_SIZE]
		padded_ENGLISH = [START_TOKEN] + padded_ENGLISH + [STOP_TOKEN] + [PAD_TOKEN] * (ENGLISH_WINDOW_SIZE - len(padded_ENGLISH)-1)
		ENGLISH_padded_sentences.append(padded_ENGLISH)

	return FRENCH_padded_sentences, ENGLISH_padded_sentences

def build_vocab(sentences):
	"""
	DO NOT CHANGE
  Builds vocab from list of sentences
	:param sentences:  list of sentences, each a list of words
	:return: tuple of (dictionary: word --> unique index, pad_token_idx)
  """
	tokens = []
	for s in sentences: tokens.extend(s)
	all_words = sorted(list(set([STOP_TOKEN,PAD_TOKEN,UNK_TOKEN] + tokens)))

	vocab =  {word:i for i,word in enumerate(all_words)}

	return vocab,vocab[PAD_TOKEN]

def convert_to_id(vocab, sentences):
	"""
	DO NOT CHANGE
  Convert sentences to indexed
	:param vocab:  dictionary, word --> unique index
	:param sentences:  list of lists of words, each representing padded sentence
	:return: numpy array of integers, with each row representing the word indeces in the corresponding sentences
  """
	return np.stack([[vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in sentence] for sentence in sentences])


def read_data(file_name):
	"""
	DO NOT CHANGE
  Load text data from file
	:param file_name:  string, name of data file
	:return: list of sentences, each a list of words split on whitespace
  """
	text = []
	with open(file_name, 'rt', encoding='latin') as data_file:
		for line in data_file: text.append(line.split())
	return text

@av.get_data_func
def get_data(french_training_file, english_training_file, french_test_file, english_test_file):
	"""
	Use the helper functions in this file to read and parse training and test data, then pad the corpus.
	Then vectorize your train and test data based on your vocabulary dictionaries.
	:param french_training_file: Path to the French training file.
	:param english_training_file: Path to the English training file.
	:param french_test_file: Path to the French test file.
	:param english_test_file: Path to the English test file.

	:return: Tuple of train containing:
	(2-d list or array with English training sentences in vectorized/id form [num_sentences x 15] ),
	(2-d list or array with English test sentences in vectorized/id form [num_sentences x 15]),
	(2-d list or array with French training sentences in vectorized/id form [num_sentences x 14]),
	(2-d list or array with French test sentences in vectorized/id form [num_sentences x 14]),
	English vocab (Dict containg word->index mapping),
	French vocab (Dict containg word->index mapping),
	English padding ID (the ID used for *PAD* in the English vocab. This will be used for masking loss)
	"""
	# MAKE SURE YOU RETURN SOMETHING IN THIS PARTICULAR ORDER: train_english, test_english, train_french, test_french, english_vocab, french_vocab, eng_padding_index

	#TODO:

	#1) Read English and French Data for training and testing (see read_data)
	french_train_data = read_data(french_training_file)
	french_test_data = read_data(french_test_file)
	english_train_data = read_data(english_training_file)
	english_test_data = read_data(english_test_file)
	#2) Pad training data (see pad_corpus)
	padded_french_train_data, padded_english_train_data= pad_corpus(french_train_data, english_train_data)
	#3) Pad testing data (see pad_corpus)
	padded_french_tests_data, padded_english_tests_data = pad_corpus(french_test_data, english_test_data)
	#4) Build vocab for French (see build_vocab)
	french_vocab, french_pad_token = build_vocab(padded_french_train_data)
	#5) Build vocab for English (see build_vocab)
	english_vocab, english_pad_token = build_vocab(padded_english_train_data)
	#6) Convert training and testing English sentences to list of IDS (see convert_to_id)
	id_french_train_data = convert_to_id(french_vocab, padded_french_train_data)
	id_french_test_data = convert_to_id(french_vocab, padded_french_tests_data)
	#7) Convert training and testing French sentences to list of IDS (see convert_to_id)
	id_english_train_data = convert_to_id(english_vocab, padded_english_train_data)
	id_english_test_data = convert_to_id(english_vocab, padded_english_tests_data)
  
	return id_english_train_data, id_english_test_data, id_french_train_data, id_french_test_data, english_vocab, french_vocab, english_pad_token


### FROM CNN PROJECT ###
def unpickle(file):
	"""
	CIFAR data contains the files data_batch_1, data_batch_2, ..., 
	as well as test_batch. We have combined all train batches into one
	batch for you. Each of these files is a Python "pickled" 
	object produced with cPickle. The code below will open up a 
	"pickled" object (each file) and return a dictionary.

	NOTE: DO NOT EDIT

	:param file: the file to unpickle
	:return: dictionary of unpickled data
	"""
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict


def get_data(file_path, first_class, second_class):
	"""
	Given a file path and two target classes, returns an array of 
	normalized inputs (images) and an array of labels. 
	You will want to first extract only the data that matches the 
	corresponding classes we want (there are 10 classes and we only want 2).
	You should make sure to normalize all inputs and also turn the labels
	into one hot vectors using tf.one_hot().
	Note that because you are using tf.one_hot() for your labels, your
	labels will be a Tensor, while your inputs will be a NumPy array. This 
	is fine because TensorFlow works with NumPy arrays.
	:param file_path: file path for inputs and labels, something 
	like 'CIFAR_data_compressed/train'
	:param first_class:  an integer (0-9) representing the first target
	class in the CIFAR10 dataset, for a cat, this would be a 3
	:param first_class:  an integer (0-9) representing the second target
	class in the CIFAR10 dataset, for a dog, this would be a 5
	:return: normalized NumPy array of inputs and tensor of labels, where 
	inputs are of type np.float32 and has size (num_inputs, width, height, num_channels) and labels 
	has size (num_examples, num_classes)
	"""
	unpickled_file = unpickle(file_path)
	inputs = unpickled_file[b'data']
	labels = unpickled_file[b'labels']
	# TODO: Do the rest of preprocessing! 
	labels = np.array(labels)
	ind_1 = np.nonzero(labels == first_class)
	ind_2 = np.nonzero(labels == second_class)
	ind = np.concatenate([ind_1[0], ind_2[0]])
	inputs = np.float32(inputs[ind]/255)
	labels = labels[ind]
	inputs = tf.reshape(inputs, (-1, 3, 32, 32))
	inputs = tf.transpose(inputs, perm=[0,2,3,1])
	labels = np.where(labels == first_class, 0, 1)
	labels = tf.one_hot(labels, 2)
	return inputs, labels
	