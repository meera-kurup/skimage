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
            # print(y_pred.shape)
            # print(batch_labels.shape)
            loss = model.loss(y_pred, batch_labels)
            # print(loss)
            model.loss_list.append(loss)
            accuracy = model.accuracy(y_pred, batch_labels)
            model.accuracy_list.append(accuracy)
        
        if b % 50 == 0:
            print("Loss after {} training steps: {}".format(b, loss))
            print("Accuracy after {} training steps: {}".format(b, accuracy))
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


def visualize_loss(losses): 
    """
    Uses Matplotlib to visualize the losses/accuracies of our model.
    :param losses: list of loss/accuracy data stored from train.

    :return: doesn't return anything, a plot should pop-up and save.
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    timestamp = time.strftime("accuracy_%Y%m%d%H%M%S.txt")
    plt.savefig('../results/loss' + timestamp + '.png')
    plt.close()

def visualize_accuracy(accuracies): 
    """
    Uses Matplotlib to visualize the accuracies of our model.
    :param accuracies: list of accuracy data stored from train.

    :return: doesn't return anything, a plot should pop-up and save.
    """
    x = [i for i in range(len(accuracies))]
    plt.plot(x, accuracies)
    plt.title('Accuracy per batch')
    plt.xlabel('Batch')
    plt.ylabel('Accuracy')
    timestamp = time.strftime("accuracy_%Y%m%d%H%M%S.txt")
    plt.savefig('../results/accuracy' + timestamp + '.png')
    plt.close()


def main():
    '''
    Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and 
    test your model for a number of epochs. We recommend that you train for
    10 epochs and at most 25 epochs. 
    
    CS1470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=70%.
    
    CS2470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=75%.
    
    :return: None
    '''
    image_size = 128
    num_classes = 5


    inputs, labels = get_data("../data/imgs.npy", "../data/labels.npy", image_size)
    #print(labels)
    # num_classes = len(label_dict)
    split = 750
    train_inputs = []
    train_labels = []
    test_inputs = []
    test_labels = []
    for n in range(num_classes):
        if n == 0 :
            train_inputs = inputs[1000*n:1000*n+split, :, :, :]
            test_inputs = inputs[1000*n+split: 1000*(n+1), :, :, :]
            train_labels = labels[1000*n:1000*n+split, :]
            test_labels = labels[1000*n+split: 1000*(n+1), :]
        else:
            train_inputs.concat(inputs[1000*n:1000*n+split, :, :, :], 0)
            test_inputs.concat(inputs[1000*n+split: 1000*(n+1), :, :, :], 0)
            train_labels.concat(labels[1000*n:1000*n+split, :], 0)
            test_labels.concat(labels[1000*n+split: 1000*(n+1), :], 0)

    # for n in range(num_classes):
    #     train_inputs[0].append(train_inputs[n])
    #     test_inputs[0].append(test_inputs[n])
    #     train_labels[0].append(train_labels[n])
    #     test_labels[0].append(test_labels[n])

    print(len(train_inputs))
    print(len(test_inputs))
    # train_inputs = np.array(train_inputs)
    # test_inputs = np.array(test_inputs)
    # train_labels = np.array(train_labels)
    # test_labels = np.array(test_labels)
    # print(train_inputs.shape)
    # print(test_inputs.shape)

    model = Model(num_classes, image_size)
    # print(len(label_dict))
    epochs = 1
    print("Training...")
    for e in range(epochs):
        print("Epoch: " + str(e+1) + "/" + str(epochs))
        train(model, train_inputs, train_labels)

    visualize_loss(model.loss_list)
    visualize_accuracy(model.accuracy_list)

    # accuracy = test(model, test_inputs, test_labels)
    # print("Model Test Average Accuracy: " + str(accuracy))

if __name__ == '__main__':
    main()