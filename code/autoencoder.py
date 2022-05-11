import tensorflow as tf
from keras.layers import Conv2D, Conv2DTranspose, GaussianNoise

class Autoencoder(tf.keras.Model):
    def __init__(self, image_size):
        super(Autoencoder, self).__init__()
        self.image_size = image_size
        self.batch_size = 64
        self.loss_list = []
        self.learning_rate = 0.001
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)
        
        #Encoder
        self.encoder = tf.keras.Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(self.image_size,self.image_size,3)),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            ])
        
        #Decoder
        self.decoder = tf.keras.Sequential([
            Conv2DTranspose(64, kernel_size=(3, 3), activation='relu'),
            Conv2DTranspose(64, kernel_size=(3, 3), activation='relu'),
            Conv2DTranspose(3, kernel_size=(3, 3), activation='relu')])
        
    def noiser(self, inputs):
        '''
        Helper function that artificially adds noise to the given inputs
        
        :param inputs: inputs
        
        :return: noised inputs
        '''
        sample = GaussianNoise(0.1)
        return sample(inputs, training=True)

    @tf.function
    def call(self, inputs):
        inputs = self.noiser(inputs)
        encoded_inputs = self.encoder(inputs)
        decoded_inputs = self.decoder(encoded_inputs)
        return decoded_inputs
        
    def loss(self, pred, labels, alpha=0.3):
        mse_loss = tf.keras.losses.MeanSquaredError() 
        bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        
        return ((1-alpha)*mse_loss(labels, pred) + alpha*bce_loss(labels, pred))
    
