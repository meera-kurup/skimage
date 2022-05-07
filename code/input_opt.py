import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D, BatchNormalization, AveragePooling2D, Input
from keras.applications.inception_v3 import InceptionV3
from keras.regularizers import l2

class InputOptimizer(tf.keras.Model):
    '''
    Optimizes inputs for a model
    '''
    def __init__(self, model, input_shape, output_shape, **kwargs):
        """
        Default constructor: Takes in a model to optimize for and example 
        input/output shapes of the model.
        """
        super().__init__(**kwargs)

        ## Save the model for which the inputs will be optimized for
        self.model = model

        self.num_probs = output_shape[1]                    ## 10
        self.opt_shape = [self.num_probs] + input_shape[1:] ## 10x28x28x1
        
        ## Set of optimizable inputs; 10 28x28 images initialized to 0.
        self.opt_input = tf.Variable(tf.zeros(self.opt_shape))  
        ## TODO: FIX THE VARIABLE INITIALIZATION (IF REQUIRED BY YOUR MODEL)

        ## Predictions to which our inputs are optimized for. nxn "eye"-dentity mtx
        self.opt_probs = tf.Variable(tf.eye(self.num_probs))

        ## Images of the optimization process to be exported as a gif
        self.opt_imgs = []


    @staticmethod
    def fig2img(fig):
        """
        Convert a Matplotlib figure to a PIL Image and return it
        https://stackoverflow.com/a/61754995/5003309
        """
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        return img


    def add_input_to_imgs(self, outputs, nrows=2, ncols=5, figsize=(10, 5)):
        '''
        Plot the image samples in outputs in a pyplot figure and add the image 
        to the 'opt_imgs' list. Used to later generate a gif. 
        '''
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for i, ax in enumerate(axs.reshape(-1)):
            ax.set_title(f'Ideal for {i}')
            out_numpy = np.squeeze(outputs[i].numpy(), -1)
            ax.imshow(out_numpy, cmap='Greys')
        self.opt_imgs += [self.fig2img(fig)]
        plt.close(fig)


    # ## FOR REFERENCE ONLY: DO NOT UNCOMMENT
    # def train_step(self, data):
    #     '''
    #     Optional Train_Step Specification; This happens for every training bach
    #     https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
    #     '''
    #     x, y = data
    #     with tf.GradientTape() as tape:
    #         y_pred = self(x, training=True)  # Forward pass
    #         # Compute the loss value (the loss function is configured in `compile()`)
    #         loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

    #     # Compute gradients
    #     gradients = tape.gradient(loss, self.trainable_variables)
    #     # Update weights
    #     self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    #     # Update metrics (includes the metric that tracks the loss)
    #     self.compiled_metrics.update_state(y, y_pred)
    #     # Return a dict mapping metric names to current value
    #     return {m.name: m.result() for m in self.metrics}


    def train(self, epochs, augment_fn=tf.identity):
        '''
        Optional Train_Step Specification; This happens for every training batch
        Uncomment this if you want to customize something
        https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
        '''
        pbar = tqdm(range(epochs))    ## Loop with nice progress bar/description
        for e in pbar: 

            ## TODO: 
            ## 1. Create a tape scope in which you augment the inputs, get the 
            ##      predictions, and compute the compiled loss.
            with tf.GradientTape() as tape:
              data = augment_fn(self.opt_input)
              y = self.opt_probs
              y_pred = self.model(data, training = True)  # Forward pass
              loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            ## 2. Optimize with the output loss with respect to the *input*
            ##      HINT: gradient and apply_gradient expect a *list* of vars...
            gradients = tape.gradient(loss, [self.opt_input])
            self.optimizer.apply_gradients(zip(gradients, [self.opt_input]))
            self.compiled_metrics.update_state(y, y_pred)
            ## Augment the images and add them to the list of output pics
            self.add_input_to_imgs(augment_fn(self.opt_input))

            ## Compute eval metric (how often optimized inputs classified right)
            out = {m.name: m.result() for m in self.metrics}
            acc = round(float(out["categorical_accuracy"]), 3)
            pbar.set_description(f'Epoch {e+1}/{epochs}: Accuracy {acc}\t')
                
        return out


## TODO: Augmentation pipeline to zooms and translates the images
augment_fn = tf.keras.Sequential([ 
          tf.keras.layers.RandomZoom(height_factor = 0.2, width_factor = 0.2),
          tf.keras.layers.RandomTranslation(height_factor = 0.2, width_factor = 0.2)
    
], name='sequential')

# Instantiate our model
input_opt_model = InputOptimizer(
    model, 
    input_shape  = X0[:,:,:,None].shape, 
    output_shape = Y0.shape
)

input_opt_model.compile(
    optimizer   = tf.keras.optimizers.Adam(learning_rate=0.05),
    loss        = tf.keras.losses.CategoricalCrossentropy(),
    metrics     = [tf.keras.metrics.CategoricalAccuracy()],
    run_eagerly = True
)

input_opt_model.train(epochs=70, augment_fn=augment_fn);