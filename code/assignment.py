import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import numpy as np
import sys
import random
import math
import time
import argparse

from matplotlib.ft2font import HORIZONTAL
from autoencoder import Autoencoder
from input_opt import InputOptimizer
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import RandomZoom, RandomTranslation, RandomRotation, RandomFlip
from preprocess import get_data
from model import Model
from matplotlib import pyplot as plt

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--autoencoder", action="store_true")
    parser.add_argument("--input_opt", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--weights", default=None, help='''Path to model weights file (should end with the extension .h5).''')
    args = parser.parse_args()
    return args

def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. Inputs 
    and labels are shuffled in the same order using tf.gather.
    
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training), 
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training), 
    shape (num_labels, num_classes)
    :return: List of losses to use for visualize_loss
    '''

    # train_inputs = tf.image.random_flip_left_right(train_inputs)
    indicies = tf.random.shuffle(range(len(train_labels)))
    train_inputs_shuffled = tf.gather(train_inputs, indicies)
    train_labels_shuffled = tf.gather(train_labels, indicies)
    num_batches = int(len(train_inputs)/model.batch_size)

    # Iterate through batches
    for b in range(num_batches):
        batch_inputs = train_inputs_shuffled[model.batch_size*b: model.batch_size*(b+1)]
        batch_inputs = tf.image.random_flip_left_right(batch_inputs)
        batch_labels = train_labels_shuffled[model.batch_size*b: model.batch_size*(b+1)]

        # Gradients
        with tf.GradientTape() as tape:
            y_pred = model.call(batch_inputs)
            if args.autoencoder:
                print(y_pred.shape)
                print(batch_labels.shape)
                loss = model.loss(y_pred, batch_inputs)
            else:
                loss = model.loss(y_pred, batch_labels)
                accuracy = model.accuracy(y_pred, batch_labels)
                model.accuracy_list.append(accuracy)
                
            model.loss_list.append(loss.numpy())

        # Print losses and accuracy (if not the autoencoder) every 50 batches
        if b % 50 == 0:
            print("Loss after {} training steps: {}".format(b, loss))
            if not args.autoencoder:
                print("Accuracy after {} training steps: {}".format(b, accuracy))
        
        # Train on gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return model.loss_list

def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels.
    
    :param test_inputs: test data (all images to be tested), 
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """
    print("Testing")
    return model.accuracy(model.call(test_inputs), test_labels)

def visualize(title, list): 
    """
    Uses Matplotlib to visualize the loss/accuracy of our model.
    
    :param title: type of data to visualize
    :param list: list of data stored from train.

    :return: doesn't return anything, a plot should pop-up and save.
    """
    x = [i for i in range(len(list))]
    print(x, len(x), len(list))
    plt.plot(x, list)
    plt.title(title + ' per batch')
    plt.xlabel('Batch')
    plt.ylabel(title)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plt.savefig('../results/' + title + "/" + title + "_" + timestamp + '.png')
    print("Saved " + title + " in " + '../results/' + title + "/" + title + "_" + timestamp + '.png')
    plt.close()
    
def undo_preprocess(input):
    """
    Undos image manipulation done in preprocessing to revert back to a standard RGB image.
    
    :param input: preprocessed inputs
    
    :return: RGB image
    """
    img = tf.dtypes.cast(input, tf.float64)
    img = tf.transpose(img, perm=[0,3,1,2])
    img = tf.reshape(img, (128, 128, 3))
    return img

def view_autoencoder_results(inputs, model, num_classes, split):
    """
    Displays the orignal inputs and outputs of the trained autoencoder

    :param inputs: preprocessed inputs
    :param model: autoencoder model
    :param num_classes: numnber of classes
    :param split: train-test split quantity

    :return: RGB image
    """
    fig = plt.figure(figsize=(8, 8))
    
    rows = 2
    columns = num_classes
    print(inputs.shape)
    for i in range(1, num_classes+1):
        original_img = inputs[i*(1000-split)-1]
        original_img = np.expand_dims(original_img, axis=0)
        
        ae_img = model.call(original_img)
        ae_img = undo_preprocess(ae_img)
        original_img = undo_preprocess(model.noiser(original_img))
        
        # add original inputs
        fig.add_subplot(rows, columns, i)
        plt.imshow(original_img)
        
        # add autoencoder outputs
        fig.add_subplot(rows, columns, i+columns)
        plt.imshow(ae_img)
        
    # save graph
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plt.savefig('../results/ae/ae_' + timestamp + '.png')
    plt.close()

def save_model_weights(model):
        """
        Save trained model weights to model_ckpts/

        Inputs:
        - model: Trained model.
        """
        output_dir = os.path.join("../model_ckpts")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, timestamp)
        os.makedirs("../model_ckpts", exist_ok=True)
        # os.makedirs(output_dir, exist_ok=True)
        model.save_weights(output_path, save_format="h5")
        print("Saved in " + output_path)

def load_weights(model, weights_path):
    """
    Load the trained model's weights.

    Inputs:
    - model: Your untrained model instance.

    Returns:
    - model: Trained model.
    """
    print("Loading from " + weights_path)

    inputs = tf.zeros([1,model.image_size,model.image_size, 3])  # Random data sample
    labels = tf.constant([[0]])

    _ = model(inputs) # Initialize trainable parameters?
    model.load_weights(args.weights) #Load weights?

    return model


def main(args):
    '''
    Read in data (limited to 5 classes), initialize model, and train and 
    test your model for a number of epochs.
    
    :return: None
    '''
    image_size = 128
    num_classes = 5

    inputs, labels = get_data("../data/imgs.npy", "../data/labels.npy", image_size)

    # Split inputs into train and test data
    split = 750
    train_inputs = []
    train_labels = []
    test_inputs = []
    test_labels = []
    for n in range(num_classes):
        if n == 0 :
            train_inputs = np.array(inputs[1000*n:1000*n+split, :, :, :])
            test_inputs = np.array(inputs[1000*n+split: 1000*(n+1), :, :, :])
            train_labels = np.array(labels[1000*n:1000*n+split, :])
            test_labels = np.array(labels[1000*n+split: 1000*(n+1), :])
        else:
            train_inputs = np.append(train_inputs, np.array(inputs[1000*n:1000*n+split, :, :, :]), axis = 0)
            test_inputs = np.append(test_inputs, np.array(inputs[1000*n+split: 1000*(n+1), :, :, :]), axis = 0)
            train_labels = np.append(train_labels, np.array(labels[1000*n:1000*n+split, :]), axis = 0)
            test_labels = np.append(test_labels, np.array(labels[1000*n+split: 1000*(n+1), :]), axis = 0)

    ### Autoencoder ###
    if args.autoencoder:
        model = Autoencoder(image_size)

        epochs = args.num_epochs
        print("Training...")
        for e in range(epochs):
            print("Epoch: " + str(e+1) + "/" + str(epochs))
            train(model, train_inputs, train_labels)
    else:
    ### CNN ###
        model = Model(num_classes, image_size)
        
        # Modify train inputs to denoise via trained autoencoder
        # ae_model = Autoencoder(image_size)
        # train_inputs = ae_model.call(train_inputs)
        
        if args.weights is None:
            # Train model
            epochs = args.num_epochs
            print("Training...")
            for e in range(epochs):
                print("Epoch: " + str(e+1) + "/" + str(epochs))
                train(model, train_inputs, train_labels)

            # Save weights
            save_model_weights(model)
        else:
            # Load weights from previous model
            load_weights(model, args.weights)

    ### Input Optimization ###
    if args.input_opt:
        # Augment function
        augment_fn = tf.keras.Sequential([ 
            RandomZoom(height_factor = 0.01, width_factor = 0.01),
            RandomTranslation(height_factor = 0.05, width_factor = 0.05),
            RandomRotation(factor=(-0.05, 0.05)),
            RandomFlip('horizontal')
            ], name='sequential')
                        
        opt_shape = (model.num_classes, model.image_size, model.image_size, 3)

        input_opt_model = InputOptimizer(
            model, 
            num_probs = model.num_classes,
            opt_shape = opt_shape
        )

        input_opt_model.compile(
            optimizer   = tf.keras.optimizers.Adam(learning_rate=0.05),
            loss        = tf.keras.losses.CategoricalCrossentropy(),
            metrics     = [tf.keras.metrics.CategoricalAccuracy()],
            run_eagerly = True
        )

        input_opt_model.train(epochs=args.num_epochs, augment_fn=augment_fn)
        # input_opt_model.train(epochs=args.num_epochs)
        imgs = input_opt_model.opt_imgs
        timestamp = time.strftime("%Y%m%d_%H%M")
        imgs[0].save('../results/input_opt/ideal_inputs_' + timestamp + '.gif', save_all=True, append_images=imgs[1:], loop=True, duration=200)
        print("Saved result as " + '../results/input_opt/ideal_inputs_' + timestamp + '.gif')
        # IPython.display.Image(open('ideal_inputs.gif','rb').read())
        
    
    # Save graphs in results folder
    if args.autoencoder:
        visualize("ae_loss", model.loss_list)
        view_autoencoder_results(test_inputs, model, num_classes, split)
    elif not args.input_opt:
        visualize("loss", model.loss_list)
        visualize("accuracy", model.accuracy_list)

        # Test model (test if weights are saving)
        accuracy = test(model, test_inputs, test_labels)
        tf.print("Model Test Average Accuracy: " + str(accuracy.numpy()))

if __name__ == '__main__':
    args = parseArguments()
    main(args)