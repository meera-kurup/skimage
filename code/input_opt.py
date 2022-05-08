import numpy as np
import tensorflow as tf
import io
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image
from tqdm import tqdm

class InputOptimizer(tf.keras.Model):
    '''
    Optimizes inputs for a model
    '''
    def __init__(self, model, num_probs, opt_shape, **kwargs):
        """
        Default constructor: Takes in a model to optimize for and example 
        input/output shapes of the model.
        """
        super().__init__(**kwargs)

        ## Save the model for which the inputs will be optimized for
        self.model = model

        self.num_probs = num_probs                   ## 10 -> 5
        self.opt_shape = opt_shape ## 10x28x28x1 -> 5 x image_size x image_size x 1
        
        ## Set of optimizable inputs; 10 28x28 images initialized to 0.
        self.opt_input = tf.Variable(tf.zeros(self.opt_shape), dtype=np.float32)
        ## FIX THE VARIABLE INITIALIZATION (IF REQUIRED BY YOUR MODEL)

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


    def add_input_to_imgs(self, outputs, nrows=1, ncols=5, figsize=(10, 5)):
        '''
        Plot the image samples in outputs in a pyplot figure and add the image 
        to the 'opt_imgs' list. Used to later generate a gif. 
        '''
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for i, ax in enumerate(axs.reshape(-1)):
            ax.set_title(f'Ideal for {i}')
            out_numpy = outputs[i].numpy()
            # out_numpy = np.squeeze(outputs[i].numpy(), -1)
            #print(out_numpy)
            # print(out_numpy.shape)
            # ax.imshow(out_numpy, vmin = -1, vmax = 1)
            # ax.imshow((out_numpy * 255).astype(np.uint8))
            out_numpy = np.clip(out_numpy, 0, 1)
            # ax.imshow(out_numpy*255)
            ax.imshow(out_numpy)
            
        self.opt_imgs += [self.fig2img(fig)]
        plt.close(fig)

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
              y_pred = self.model(data)  # Forward pass
              #print(y_pred)
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
# augment_fn = tf.keras.Sequential([ 
#           tf.keras.layers.RandomZoom(height_factor = 0.2, width_factor = 0.2),
#           tf.keras.layers.RandomTranslation(height_factor = 0.2, width_factor = 0.2)
    
# ], name='sequential')


# augment_fn = ImageDataGenerator(featurewise_center=False,
#                  samplewise_center=False,
#                  featurewise_std_normalization=False,
#                  samplewise_std_normalization=False,
#                  zca_whitening=False,
#                  rotation_range=5,
#                  width_shift_range=0.05,
#                  height_shift_range=0.05,
#                  shear_range=0.2,
#                  zoom_range=0.2,
#                  channel_shift_range=0.,
#                  fill_mode='nearest',
#                  cval=0.,
#                  horizontal_flip=True,
#                  vertical_flip=False,
#                  rescale=1/255) #rescale to [0-1], add zoom range of 0.2x and horizontal flip

