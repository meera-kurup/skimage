# SKIMage

SKIMage: Cleverly Classifying Culinary Cuisines

Devpost: https://devpost.com/software/creatively-captioning-culinary-cuisines

# Model Architecture

We used a simplified version of the model in the paper "Food Classification from Images Using Convolutional Neural Networks" due to limitations in computing power. We experimented with multiple model architectures and reached the following layer structure that had relatively high accuracy:

```
Conv2D(32, kernel_size=(3, 3), activation='relu')
MaxPooling2D(pool_size=(2, 2))
Conv2D(64, kernel_size=(3, 3), activation='relu')
MaxPooling2D(pool_size=(2, 2))
Flatten()
Dense(64, activation='relu')
Dense(32, activation='relu')
Dense(self.num_classes, activation='softmax')
```

# Running model

To run, type the following into the terminal

```
python assignment.py
```

Options and arguments:

```
--autoencoder   : Runs the autoencoder
--input_opt     : Runs the input optimizer along with the model
--learning_rate : Changes the learning rate (default is 0.001)
--num_epochs    : Changes number of epochs (default is 50)
--weights arg   : Loads in model weights (must end in .h5 extension); arg is path to weights
```
