import os
import shutil

import numpy as np

# A quick and dirty script to create a tefla compatible data dir
# Use as a starting point for your own, as needed
dest_dir = '/home/lalit/work/dogscats/data.dbg/'
num_training_images_per_category = 1000
num_validation_images_per_category = 200

# training
path_prefix = '/home/lalit/work/dogscats/raw_data/train/'
cats_dir = path_prefix + 'cats/'
dogs_dir = path_prefix + 'dogs/'

training_dir = dest_dir + 'training/'
if os.path.exists(dest_dir):
    shutil.rmtree(dest_dir)

os.makedirs(training_dir)
cats = os.listdir(cats_dir)[:num_training_images_per_category]
dogs = os.listdir(dogs_dir)[:num_training_images_per_category]

for c in cats:
    os.symlink(cats_dir + c, training_dir + c)

for d in dogs:
    os.symlink(dogs_dir + d, training_dir + d)

cat_names = [c[:-4] for c in cats]
dog_names = [d[:-4] for d in dogs]
training_images = cat_names + dog_names

cat_labels = [0 for _ in cats]
dog_labels = [1 for _ in dogs]
training_labels = cat_labels + dog_labels

header = ['image', 'label']

out = np.column_stack((training_images, training_labels))
out = np.row_stack((header, out))
np.savetxt(dest_dir + 'training_labels.csv', out, delimiter=',', fmt='%s')

# validation
path_prefix = '/home/lalit/work/dogscats/raw_data/valid/'
cats_dir = path_prefix + 'cats/'
dogs_dir = path_prefix + 'dogs/'

validation_dir = dest_dir + 'validation/'
os.makedirs(validation_dir)
cats = os.listdir(cats_dir)[:num_validation_images_per_category]
dogs = os.listdir(dogs_dir)[:num_validation_images_per_category]

for c in cats:
    os.symlink(cats_dir + c, validation_dir + c)

for d in dogs:
    os.symlink(dogs_dir + d, validation_dir + d)

cat_names = [c[:-4] for c in cats]
dog_names = [d[:-4] for d in dogs]
validation_images = cat_names + dog_names

cat_labels = [0 for _ in cats]
dog_labels = [1 for _ in dogs]
validation_labels = cat_labels + dog_labels

header = ['image', 'label']

out = np.column_stack((validation_images, validation_labels))
out = np.row_stack((header, out))
np.savetxt(dest_dir + 'validation_labels.csv', out, delimiter=',', fmt='%s')
