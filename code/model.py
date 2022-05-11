import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, AveragePooling2D, Input
from keras.applications.inception_v3 import InceptionV3
from keras.regularizers import l2

class Model(tf.keras.Model):
    """
    A deep learning model class that conatins the architecture
    for our CNN that classifies images.
    """
    def __init__(self, num_classes, image_size):
        super(Model, self).__init__()
        
        ### HYPERPARAMETERS ###
        self.num_classes = num_classes
        self.batch_size = 64
        self.loss_list = []
        self.accuracy_list = []
        self.learning_rate = 0.0001
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)
        self.image_size = image_size
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


        # ## CONV ##
        # self.cnn = tf.keras.Sequential([
        #     Conv2D(filters = 32, kernel_size = (5,5), strides = 2, padding = 'Same', activation ='relu'),
        #     Conv2D(filters = 32, kernel_size = (5,5), strides = 2, padding = 'Same', activation ='relu'),
        #     MaxPool2D(pool_size=(2,2)),
        #     BatchNormalization(),
        #     Dropout(0.2),

        #     Conv2D(filters = 64, kernel_size = (3,3), strides = 2, padding = 'Same', activation ='relu'),
        #     Conv2D(filters = 64, kernel_size = (3,3), strides = 2, padding = 'Same', activation ='relu'), 
        #     MaxPool2D(pool_size=(2,2)),
        #     BatchNormalization(),
        #     Dropout(0.2),

        #     Conv2D(filters = 128, kernel_size = (2,2),padding = 'Same', activation ='relu'),
        #     Conv2D(filters = 128, kernel_size = (2,2),padding = 'Same', activation ='relu'),
        #     MaxPool2D(pool_size=(2,2)),
        #     BatchNormalization(),
        #     Dropout(0.2),

        #     Conv2D(filters = 256, kernel_size = (2,2),padding = 'Same', activation ='relu'),
        #     Conv2D(filters = 256, kernel_size = (2,2),padding = 'Same', activation ='relu'),
        #     GlobalAveragePooling2D(),
        #     Dense(512, activation="relu"),
        #     Dropout(0.2),
        #     Dense(self.num_classes, activation="softmax", kernel_regularizer=l2()),
        # ])

        ### Final Model ###
        self.cnn = tf.keras.Sequential()
        self.cnn.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(self.image_size,self.image_size,3)))
        self.cnn.add(MaxPooling2D(pool_size=(2, 2)))
        self.cnn.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.cnn.add(MaxPooling2D(pool_size=(2, 2)))
        self.cnn.add(Flatten())
        self.cnn.add(Dense(64, activation='relu'))
        self.cnn.add(Dense(32, activation='relu'))
        self.cnn.add(Dense(self.num_classes, activation='softmax'))

    @tf.function
    def call(self, input):
        """
        Runs a forward pass on an input batch of images
        """
        # Running forward pass
        output = self.cnn(input)
        return output
    
    def accuracy(self, prbs, labels):
        """
        Calculates the model's prediction accuracy by comparing
        prbs to correct labels
        """
        correct_predictions = tf.equal(tf.argmax(prbs, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    
    def loss(self, prbs, labels):
        """
        Calculates the model categorical cross-entropy loss
        after one forward pass
        """
        # loss = tf.nn.softmax_cross_entropy_with_logits(labels, prbs)
        # avg_loss = tf.reduce_mean(loss)
        # return avg_loss
        return tf.reduce_sum(tf.keras.losses.categorical_crossentropy(labels, prbs))
    
    