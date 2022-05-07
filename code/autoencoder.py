import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D, BatchNormalization, AveragePooling2D, Input
from keras.applications.inception_v3 import InceptionV3
from keras.regularizers import l2

class Autoencoder(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        ## TODO: Implement call function
        inputs = self.encoder(inputs)
        inputs = self.decoder(inputs)
        return inputs

## Some common keyword arguments you way want to use. HINT: func(**kwargs)
conv_kwargs = { 
    "padding"             : "SAME", 
    "activation"          : tf.keras.layers.LeakyReLU(alpha=0.2), 
    "kernel_initializer"  : tf.random_normal_initializer(stddev=.1)
}

## TODO: Make encoder and decoder sub-models
ae_model = Autoencoder(
    encoder = tf.keras.Sequential([
        Conv2D(10,3, strides=2, **conv_kwargs),
        Conv2D(10,3, strides=2, **conv_kwargs),
        Conv2D(10,3, strides=2, **conv_kwargs)
    ], name='ae_encoder'),
    decoder = tf.keras.Sequential([
        Conv2DTranspose(10,3, strides=2, **conv_kwargs),
        Conv2DTranspose(10,3, strides=2, **conv_kwargs),
        Conv2DTranspose(1,3, strides=2, **conv_kwargs)
    ], name='ae_decoder')
, name='autoencoder')

ae_model.build(input_shape = X0.shape)   ## Required to see architecture summary
initial_weights = ae_model.get_weights() ## Just so we can reset out autoencoder

ae_model.summary()
ae_model.encoder.summary()
ae_model.decoder.summary()