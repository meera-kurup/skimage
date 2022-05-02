import os
from PIL import Image
import numpy as np

train_labels = []
train_images = []

test_labels = []
test_images = []
img_len = 299
maxsize = (img_len, img_len)

print("Reading training data...")
for subdir, dirs, files in os.walk("../data/train/"):
    for file in files:
        train_label = subdir.split('/')[-1]
        train_labels.append(train_label)
        i = Image.open(os.path.join(subdir, file))
        i.thumbnail(maxsize, Image.ANTIALIAS)
        
        width, height = i.size   # Get dimensions
        left = (width - img_len)/2
        top = (height - img_len)/2
        right = (width + img_len)/2
        bottom = (height + img_len)/2

        # Crop the center of the image
        i = i.crop((left, top, right, bottom))
        train_images.append(np.array(i))
        # print(label, np.array(i))
        i.close()
print("Saving training data...")
np.save('../data/train_labels', np.array(train_labels))
np.save('../data/train_data', np.array(train_images))
train_labels = None
train_iamges = None

print("Reading testing data...")
for subdir, dirs, files in os.walk("../data/test/"):
    for file in files:
        test_label = subdir.split('/')[-1]
        test_labels.append(test_label)
        i = Image.open(os.path.join(subdir, file))
        i.thumbnail(maxsize, Image.ANTIALIAS)
        
        width, height = i.size   # Get dimensions
        left = (width - img_len)/2
        top = (height - img_len)/2
        right = (width + img_len)/2
        bottom = (height + img_len)/2

        # Crop the center of the image
        i = i.crop((left, top, right, bottom))
        test_images.append(np.array(i))
        # print(label, np.array(i))
        i.close()

print("Saving testing data...")
np.save('../data/test_labels', np.array(test_labels))
np.save('../data/test_data', np.array(test_images))
print("DONE")