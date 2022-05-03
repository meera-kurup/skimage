import os
from PIL import Image
import numpy as np

labels = []
images = []

# test_labels = []
# test_images = []
img_len = 32
maxsize = (img_len, img_len)

print("Reading data...")
count = 0
for subdir, dirs, files in os.walk("../data/images/"):
    for file in files:
        label = subdir.split('/')[-1]
        labels.append(label)
        i = Image.open(os.path.join(subdir, file))
        # i.thumbnail(maxsize, Image.ANTIALIAS)

        width, height = i.size   # Get dimensions
        left = (width - img_len)/2
        top = (height - img_len)/2
        right = (width + img_len)/2
        bottom = (height + img_len)/2

        # Crop the center of the image
        i_arr = np.array(i.crop((left, top, right, bottom)))
        # print(i_arr.shape)
        if len(i_arr.shape) != 3 or i_arr.shape[2] != 3:
            i_arr = np.stack((i_arr[0],)*3, axis=-1)
            print(os.path.join(subdir, file))
        images.append(i_arr)
        # print(label, np.array(i))
        i.close()
        count += 1
    if count % 1000 == 0:
        print("Processed " + str(count) + " Images")
print("Saving data...")
np.save('../data/labels', np.array(labels))
np.save('../data/imgs', np.array(images))
train_labels = None
train_iamges = None
print("DONE")