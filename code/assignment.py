from autoencoder import Autoencoder
from preprocess import get_data
from model import Model
from matplotlib import pyplot as plt
import os
import tensorflow as tf
import numpy as np
import sys
import random
import math
import time
import argparse

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_weights", action="store_true")
    # parser.add_argument("--batch_size", type=int, default=128)
    # parser.add_argument("--num_epochs", type=int, default=50)
    # parser.add_argument("--image_size", type=int, default=128*128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    args = parser.parse_args()
    return args

def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs 
    and labels - ensure that they are shuffled in the same order using tf.gather or zipping.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.
    
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training), 
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training), 
    shape (num_labels, num_classes)
    :return: Optionally list of losses per batch to use for visualize_loss
    '''

    # train_inputs = tf.image.random_flip_left_right(train_inputs)
    indicies = tf.random.shuffle(range(len(train_labels)))
    train_inputs_shuffled = tf.gather(train_inputs, indicies)
    train_labels_shuffled = tf.gather(train_labels, indicies)
    num_batches = int(len(train_inputs)/model.batch_size)

    for b in range(num_batches):
        batch_inputs = train_inputs_shuffled[model.batch_size*b: model.batch_size*(b+1)]
        batch_inputs = tf.image.random_flip_left_right(batch_inputs)
        batch_labels = train_labels_shuffled[model.batch_size*b: model.batch_size*(b+1)]

        with tf.GradientTape() as tape:
            y_pred = model.call(batch_inputs)
            print(y_pred.shape)
            print(batch_labels.shape)
            # print(batch_labels.shape)
            loss = model.loss(y_pred, batch_labels)
            model.loss_list.append(loss)
            # accuracy = model.accuracy(y_pred, batch_labels)
            # model.accuracy_list.append(accuracy)
        
        if b % 50 == 0:
            print("Loss after {} training steps: {}".format(b, loss))
            # print("Accuracy after {} training steps: {}".format(b, accuracy))
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return model.loss_list

def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly 
    flip images or do any extra preprocessing.
    
    :param test_inputs: test data (all images to be tested), 
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """
    # model_accuracy = 0
    # print("Entering Test")

    # for i in range(0, len(test_inputs), model.batch_size):
    #     input_batches = test_inputs[i:i+model.batch_size,:]
    #     label_batches = test_labels[i:i+model.batch_size,:]
    #     logits = model.call(input_batches)
    #     model_accuracy += model.accuracy(logits, label_batches)
    # batch_num = int(len(test_inputs)) / model.batch_size
    # avg_accuracy = float(model_accuracy/batch_num)

    return model.accuracy(model.call(test_inputs), test_labels)

def visualize(title, list): 
    """
    Uses Matplotlib to visualize the loss/accuracy of our model.
    :param title: type of data to visualize
    :param list: list of data stored from train.

    :return: doesn't return anything, a plot should pop-up and save.
    """
    x = [i for i in range(len(list))]
    plt.plot(x, list)
    plt.title(title + ' per batch')
    plt.xlabel('Batch')
    plt.ylabel(title)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plt.savefig('../results/' + title + "_" + timestamp + '.png')
    plt.close()

def save_model_weights(model, args):
        """
        Save trained model weights to model_ckpts/

        Inputs:
        - model: Trained model.
        - args: All arguments.
        """

        output_path = os.path.join("model_ckpts")
        os.makedirs("model_ckpts", exist_ok=True)
        model.save_weights(output_path)

def load_weights(model):
    """
    Load the trained model's weights.

    Inputs:
    - model: Your untrained model instance.

    Returns:
    - model: Trained model.
    """

    num_classes = model.num_classes

    inputs = tf.zeros([1,1,model.image_size,model.image_size])  # Random data sample
    labels = tf.constant([[0]])

    weights_path = os.path.join("model_ckpts")
    _ = model(inputs) # Initialize trainable parameters?
    model.load_weights(weights_path) #Load weights?

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

    # Train model
    model = Model(num_classes, image_size)
    autoencoder = Autoencoder(image_size)
    # autoencoder.build(input_shape = (64, 128, 128, 3)) 
    # autoencoder.summary()
    # autoencoder.encoder.summary()
    # autoencoder.decoder.summary()
    epochs = 50
    print("Training...")
    for e in range(epochs):
        print("Epoch: " + str(e+1) + "/" + str(epochs))
        train(autoencoder, train_inputs, train_labels)

    # Save graphs in results folder
    visualize("loss", model.loss_list)
    # visualize("accuracy", model.accuracy_list)

    # Test model
    # accuracy = test(model, test_inputs, test_labels)
    # tf.print("Model Test Average Accuracy: " + str(accuracy.numpy()))

        # Load trained weights
    # if args.load_weights:
    #     model = load_weights(model)
    # else:
    #     epochs = 50
    #     print("Training...")
    #     for e in range(epochs):
    #         print("Epoch: " + str(e+1) + "/" + str(epochs))
    #         train(model, train_inputs, train_labels)

    # Save graphs in results folder
    # visualize("loss", model.loss_list)
    # visualize("accuracy", model.accuracy_list)

    # Test model
    # accuracy = test(model, test_inputs, test_labels)
    # tf.print("Model Test Average Accuracy: " + str(accuracy.numpy()))

    save_model_weights(model, args)

if __name__ == '__main__':
    args = parseArguments()
    main(args)