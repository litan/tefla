"""A dataset based on features extracted from files in a training and validation directory,
and corresponding label files."""
from __future__ import division, print_function, absolute_import

import logging

import numpy as np

from tefla.core import data_load_ops as data

logger = logging.getLogger('tefla')


class DataSet(object):
    def __init__(self, data_dir, img_size):
        self.data_dir = data_dir
        training_images_dir = "%s/training_%d" % (data_dir, img_size)
        training_labels_file = "%s/training_labels.csv" % data_dir
        training_features_file = "%s/training_features.npy" % data_dir

        validation_images_dir = "%s/validation_%d" % (data_dir, img_size)
        validation_labels_file = "%s/validation_labels.csv" % data_dir
        validation_features_file = "%s/validation_features.npy" % data_dir

        self._training_files = data.get_image_files(training_images_dir)
        names = data.get_names(self._training_files)
        self._training_labels = data.get_labels(names, label_file=training_labels_file).astype(np.int32)
        self._training_features = np.load(training_features_file)

        self._validation_files = data.get_image_files(validation_images_dir)
        names = data.get_names(self._validation_files)
        self._validation_labels = data.get_labels(names, label_file=validation_labels_file).astype(np.int32)
        self._validation_features = np.load(validation_features_file)

    @property
    def training_X(self):
        return self._training_features

    @property
    def training_y(self):
        return self._training_labels

    @property
    def validation_X(self):
        return self._validation_features

    @property
    def validation_y(self):
        return self._validation_labels

    def num_training_features(self):
        return len(self._training_features)

    def num_validation_features(self):
        return len(self._validation_features)

    def num_classes(self):
        return len(np.unique(self._training_labels))

    def class_frequencies(self):
        return zip(np.unique(self._training_labels), np.bincount(self._training_labels))

    def print_info(self):
        logger.info("Data: #Training features: %d" % self.num_training_features())
        logger.info("Data: #Classes: %d" % self.num_classes())
        logger.info("Data: Class frequencies: %s" % self.class_frequencies())
        logger.info("Data: #Validation features: %d" % self.num_validation_features())
