import os
from PIL import Image
import numpy as np

### THIS IS A SCRIPT FOR READING IMAGES FROM DATASET ###

labels = []
images = []
food_list = ["sashimi", "baklava", "ramen", "edamame", "chocolate_cake"]

# test_labels = []
# test_images = []
img_len = 128
maxsize = (img_len, img_len)

print("Reading data...")
count = 0
for subdir, dirs, files in os.walk("../../../meerakurup/images/"):
    print(subdir)
    if subdir.split('/')[-1] in food_list:
        for file in files:
            if file != ".DS_Store":
                label = subdir.split('/')[-1]
                labels.append(label)
                i = Image.open(os.path.join(subdir, file))
                i.thumbnail(maxsize, Image.ANTIALIAS)
                
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
# train_labels = None
# train_iamges = None

# print("Reading testing data...")
# for subdir, dirs, files in os.walk("../data/test/"):
#     for file in files:
#         test_label = subdir.split('/')[-1]
#         test_labels.append(test_label)
#         i = Image.open(os.path.join(subdir, file))
#         i.thumbnail(maxsize, Image.ANTIALIAS)
        
#         width, height = i.size   # Get dimensions
#         left = (width - img_len)/2
#         top = (height - img_len)/2
#         right = (width + img_len)/2
#         bottom = (height + img_len)/2

#         # Crop the center of the image
#         i = i.crop((left, top, right, bottom))
#         test_images.append(np.array(i))
#         # print(label, np.array(i))
#         i.close()

# print("Saving testing data...")
# np.save('../data/test_labels', np.array(test_labels))
# np.save('../data/test_data', np.array(test_images))
print("DONE")