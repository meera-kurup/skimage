import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D, BatchNormalization, AveragePooling2D, Input
from keras.applications.inception_v3 import InceptionV3

class Model(tf.keras.Model):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        
        ### HYPERPARAMETERS ###
        self.num_classes = num_classes
        self.batch_size = 64
        self.loss_list = []
        self.learning_rate = 0.0001
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)
        image_size = 256
        # self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)

        # ### CNN from New Paper ###
        # self.cnn = tf.keras.Sequential([
        #     InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(image_size,image_size,3))),
        #     AveragePooling2D(),
        #     Dropout(0.4),
        #     Flatten(),
        #     Dense(self.num_classes)
        # ])

        
        ### CNN ###
        # self.cnn = tf.keras.Sequential([
        #     Conv2D(16, (3, 3), input_shape=(32, 32, 3), activation='relu'),
        #     MaxPool2D(pool_size=(2,2)),
        #     Conv2D(16, (3, 3), activation = 'relu'),
        #     MaxPool2D(pool_size=(2,2)),
        #     Flatten(),
        #     Dense(128, activation='relu'),
        #     Dense(num_classes)])
        # 1D or 2D Max pooling?
        # train for more epochs

        #     # CONVOLUTION LAYERS
        #     # InceptionV3(weights='imagenet', include_top=False, input_tensor=(299,299,3)),

        ## BEST ###
        self.cnn = tf.keras.Sequential([
            Conv2D(filters = 32, kernel_size = (5,5), strides = 2, padding = 'Same', activation ='relu'),
            Conv2D(filters = 32, kernel_size = (5,5), strides = 2, padding = 'Same', activation ='relu'),
            MaxPool2D(pool_size=(2,2)),
            BatchNormalization(),
            Dropout(0.2),

            Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'),
            Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'), 
            MaxPool2D(pool_size=(2,2)),
            BatchNormalization(),
            Dropout(0.2),

            Conv2D(filters = 128, kernel_size = (2,2),padding = 'Same', activation ='relu'),
            Conv2D(filters = 128, kernel_size = (2,2),padding = 'Same', activation ='relu'),
            MaxPool2D(pool_size=(2,2)),
            BatchNormalization(),
            Dropout(0.2),

            Conv2D(filters = 256, kernel_size = (2,2),padding = 'Same', activation ='relu'),
            Conv2D(filters = 256, kernel_size = (2,2),padding = 'Same', activation ='relu'),
            GlobalAveragePooling2D(),
            Dense(512, activation="relu"),
            Dropout(0.2),
            # Flatten(),
            Dense(self.num_classes),
            # Dropout(0.2),
            # Dense(self.num_classes)
        ])

        # self.cnn = tf.keras.Sequential([
        #     Conv2D(filters = 128, kernel_size = (5,5), strides = 2, padding = 'Same', activation ='relu'),
        #     Conv2D(filters = 128, kernel_size = (5,5), strides = 2, padding = 'Same', activation ='relu'),
        #     MaxPool2D(pool_size=(2,2)),
        #     BatchNormalization(),
        #     Dropout(0.4),

        #     Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'),
        #     Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'), 
        #     MaxPool2D(pool_size=(2,2)),
        #     BatchNormalization(),
        #     Dropout(0.4),

        #     Conv2D(filters = 32, kernel_size = (2,2),padding = 'Same', activation ='relu'),
        #     Conv2D(filters = 32, kernel_size = (2,2),padding = 'Same', activation ='relu'),
        #     MaxPool2D(pool_size=(2,2)),
        #     BatchNormalization(),
        #     Dropout(0.4),

        #     GlobalAveragePooling2D(),
        #     Flatten(),
        #     Dense(512, activation='relu'),
        #     Dropout(0.4),
        #     # Flatten(),
        #     # Dense(256, activation='relu'),
        #     # Dropout(0.4),
        #     # Flatten(),
        #     # Dense(64, activation='relu'),
        #     # Dropout(0.4),
        #     Dense(self.num_classes)
        # ])

        # self.cnn = tf.keras.Sequential()
        # self.cnn.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32,32,3)))
        # self.cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        # self.cnn.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
        # self.cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        # self.cnn.add(tf.keras.layers.Flatten())
        # self.cnn.add(tf.keras.layers.Dense(256, activation='relu'))
        # self.cnn.add(tf.keras.layers.Dense(128, activation='relu'))
        # self.cnn.add(tf.keras.layers.Dense(self.num_classes))

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
    
    