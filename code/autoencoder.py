import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten, Conv2D,  MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, AveragePooling2D, Input, Conv2DTranspose
from keras.applications.inception_v3 import InceptionV3
from keras.regularizers import l2

class Autoencoder(tf.keras.Model):
    def __init__(self, image_size):
        super(Autoencoder, self).__init__()
        self.image_size = image_size
        self.batch_size = 64
        self.loss_list = []
        self.learning_rate = 0.001
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)
        self.encoder = tf.keras.Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(self.image_size,self.image_size,3)),
            # MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            # MaxPooling2D(pool_size=(2, 2))
            ])
        
        self.decoder = tf.keras.Sequential([
            # MaxUnpooling2D(pool_size=(2, 2)),
            Conv2DTranspose(64, kernel_size=(3, 3), activation='relu'),
            # MaxUnpooling2D(pool_size=(2, 2)),
            Conv2DTranspose(3, kernel_size=(3, 3), activation='relu')])

    @tf.function
    def call(self, inputs):
        encoded_inputs = self.encoder(inputs)
        decoded_inputs = self.decoder(encoded_inputs)
        return decoded_inputs
        
    def loss(self, pred, labels, alpha=0.3):
        mse_loss = tf.keras.losses.MeanSquaredError() 
        bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        
        return (1-alpha)*mse_loss(labels, pred) + alpha*bce_loss(labels, pred)
