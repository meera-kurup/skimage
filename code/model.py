import numpy as np
import tensorflow as tf

class Model(tf.keras.Model):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        
        ### HYPERPARAMETERS ###
        self.num_classes = num_classes
        self.batch_size = 100
        self.loss_list = []
        self.learning_rate = 0.001
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)
        
        ### CNN ###
        # self.cnn = tf.keras.Sequential([
        #     # CONVOLUTION LAYERS
        #     tf.keras.layers.Conv2D(64, (3, 3), padding = "same", activation='relu'),
        #     tf.keras.layers.MaxPooling2D((2, 2)),
        #     tf.keras.layers.Dropout(0.2),

        #     tf.keras.layers.Conv2D(128, (3, 3), padding = "same", activation='relu'),
        #     tf.keras.layers.Conv2D(128, (3, 3), padding = "same", activation='relu'),
        #     tf.keras.layers.MaxPooling2D((2, 2)),
        #     tf.keras.layers.Dropout(0.2),

        #     tf.keras.layers.Conv2D(256, (3, 3), padding = "same", activation='relu'),
        #     tf.keras.layers.Conv2D(256, (3, 3), padding = "same", activation='relu'),
        #     tf.keras.layers.Conv2D(256, (3, 3), padding = "same", activation='relu'),
        #     tf.keras.layers.MaxPooling2D((2,2)),
        #     tf.keras.layers.Dropout(0.2),
            
        #     # tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            
        #     # tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
           
        #     # tf.keras.layers.Conv2D(32, (5, 5), activation='relu'),
        #     # tf.keras.layers.Conv2D(32, (5, 5), activation='relu'),
        #     # tf.keras.layers.GlobalAveragePooling2D(),

        #     # DENSE LAYERS
        #     tf.keras.layers.Flatten(),
        #     tf.keras.layers.Dense(100, activation='relu'),
        #     tf.keras.layers.Dropout(0.2),
        #     # tf.keras.layers.Dense(60, activation='relu'),
        #     # tf.keras.layers.Dropout(0.2),
        #     # tf.keras.layers.Dense(30, activation='relu'),
        #     # tf.keras.layers.Dropout(0.2),
        #     # tf.keras.layers.Dense(10, activation='relu'),
        #     # tf.keras.layers.Dropout(0.2),
        #     tf.keras.layers.Dense(self.num_classes, activation='softmax')
        # ])
        self.cnn = tf.keras.Sequential()
        self.cnn.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32,32,3)))
        self.cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.cnn.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.cnn.add(tf.keras.layers.Flatten())
        self.cnn.add(tf.keras.layers.Dense(256, activation='relu'))
        self.cnn.add(tf.keras.layers.Dense(128, activation='relu'))
        self.cnn.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))
        
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
    
    