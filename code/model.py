import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D, BatchNormalization, AveragePooling2D
from keras.applications.inception_v3 import InceptionV3

class Model(tf.keras.Model):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        
        ### HYPERPARAMETERS ###
        self.num_classes = num_classes
        self.batch_size = 75
        self.loss_list = []
        self.learning_rate = 0.001
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)

        # ### CNN from New Paper ###
        # self.cnn = tf.keras.Sequential([
        #     InceptionV3(weights='imagenet', include_top=False, input_tensor=(299,299,3)),
        #     AveragePooling2D(pool_size=(8,8)),
        #     Conv2D(filters = 299, kernel_size = (299,3),padding = 'Same', activation ='relu'),
        #     MaxPool2D(pool_size=(2,2)),
        #     #concat
        #     Dropout(0.4),
        #     #fully connected 
        #     Dense(self.num_classes, activation = "softmax")
        # ])

        
        ### CNN ###
        self.cnn = tf.keras.Sequential([
            # Conv2D(16, (3, 3), input_shape=(32, 32, 3), activation='relu'),
            # MaxPool2D(pool_size=(2,2)),
            # Conv2D(16, (3, 3), activation = 'relu'),
            # MaxPool2D(pool_size=(2,2)),

            # Flatten(),
            # Dense(128, activation='relu'),
            # Dense(num_classes, activation = 'softmax')

            # CONVOLUTION LAYERS
            # InceptionV3(weights='imagenet', include_top=False, input_tensor=(299,299,3)),

            Conv2D(filters = 32, kernel_size = (5,5), strides = 2, padding = 'Same', activation ='relu'),
            Conv2D(filters = 32, kernel_size = (5,5), strides = 2, padding = 'Same', activation ='relu'),
            MaxPool2D(pool_size=(2,2)),
            BatchNormalization(),
            Dropout(0.4),

            Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'),
            Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'), 
            MaxPool2D(pool_size=(2,2)),
            BatchNormalization(),
            Dropout(0.4),

            Conv2D(filters = 128, kernel_size = (2,2),padding = 'Same', activation ='relu'),
            Conv2D(filters = 128, kernel_size = (2,2),padding = 'Same', activation ='relu'),
            MaxPool2D(pool_size=(2,2)),
            BatchNormalization(),
            Dropout(0.4),

            GlobalAveragePooling2D(),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.4),
            Dense(self.num_classes, activation = "softmax"),
        ])
        
    @tf.function
    def call(self, input):
        # Running forward pass
        output = self.cnn(input)
        return output
    
    def accuracy(self, prbs, labels):
        correct_predictions = tf.equal(tf.argmax(prbs, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    
    def loss(self, prbs, labels):
        loss = tf.nn.softmax_cross_entropy_with_logits(labels, prbs)
        avg_loss = tf.reduce_mean(loss)
        return avg_loss
    
    