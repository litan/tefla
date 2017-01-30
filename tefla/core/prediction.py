from __future__ import division, print_function, absolute_import

import time

import numpy as np
import tensorflow as tf
from scipy.stats.mstats import gmean

from tefla.da import tta
from tefla.da.iterator import BatchIterator
from tefla.utils import util


class PredictSessionMixin(object):
    def __init__(self, weights_from):
        self.weights_from = weights_from

    def predict(self, X):
        graph = tf.Graph()
        with graph.as_default():
            self._build_model()
            saver = tf.train.Saver()
            with tf.Session() as sess:
                print('Loading weights from: %s' % self.weights_from)
                util.load_variables(sess, saver, self.weights_from)
                return self._real_predict(X, sess)

    def _real_predict(self, X, sess):
        pass

    def _build_model(self):
        pass


class OneCropPredictor(PredictSessionMixin):
    def __init__(self, model, cnf, weights_from, prediction_iterator, output_layer='predictions'):
        self.model = model
        self.output_layer = output_layer
        self.cnf = cnf
        self.prediction_iterator = prediction_iterator
        super(OneCropPredictor, self).__init__(weights_from)

    def _build_model(self):
        end_points_predict = self.model(is_training=False, reuse=None)
        self.inputs = end_points_predict['inputs']
        self.predictions = end_points_predict[self.output_layer]

    def _real_predict(self, X, sess, xform=None, crop_bbox=None):
        tic = time.time()
        print('Making %d predictions' % len(X))
        data_predictions = []
        for X, y in self.prediction_iterator(X, xform=xform, crop_bbox=crop_bbox):
            predictions_e = sess.run(self.predictions, feed_dict={self.inputs: X})
            data_predictions.append(predictions_e)
        data_predictions = np.vstack(data_predictions)
        print('took %6.1f seconds' % (time.time() - tic))
        return data_predictions


class InputFeaturesPredictor(PredictSessionMixin):
    def __init__(self, model, cnf, weights_from, output_layer='predictions'):
        self.model = model
        self.output_layer = output_layer
        self.cnf = cnf
        self.prediction_iterator = BatchIterator(cnf['batch_size_test'], False)
        super(InputFeaturesPredictor, self).__init__(weights_from)

    def _build_model(self):
        end_points_predict = self.model(is_training=False, reuse=None)
        self.inputs = end_points_predict['inputs']
        self.predictions = end_points_predict[self.output_layer]

    def _real_predict(self, X, sess):
        tic = time.time()
        print('Making %d predictions' % len(X))
        data_predictions = []
        for X, y in self.prediction_iterator(X):
            predictions_e = sess.run(self.predictions, feed_dict={self.inputs: X})
            data_predictions.append(predictions_e)
        data_predictions = np.vstack(data_predictions)
        print('took %6.1f seconds' % (time.time() - tic))
        return data_predictions


class QuasiCropPredictor(PredictSessionMixin):
    def __init__(self, model, cnf, weights_from, prediction_iterator, number_of_transforms, output_layer='predictions'):
        self.number_of_transforms = number_of_transforms
        self.cnf = cnf
        self.prediction_iterator = prediction_iterator
        self.predictor = OneCropPredictor(model, cnf, weights_from, prediction_iterator, output_layer)
        super(QuasiCropPredictor, self).__init__(weights_from)

    def _build_model(self):
        self.predictor._build_model()

    def _real_predict(self, X, sess):
        standardizer = self.prediction_iterator.standardizer
        da_params = standardizer.da_processing_params()
        util.veryify_args(da_params, ['sigma'], 'QuasiPredictor.standardizer does unknown da with param(s):')
        color_sigma = da_params.get('sigma', 0.0)
        tfs, color_vecs = tta.build_quasirandom_transforms(self.number_of_transforms, color_sigma=color_sigma,
                                                           **self.cnf['aug_params'])
        multiple_predictions = []
        for i, (xform, color_vec) in enumerate(zip(tfs, color_vecs), start=1):
            print('Quasi-random tta iteration: %d' % i)
            standardizer.set_tta_args(color_vec=color_vec)
            predictions = self.predictor._real_predict(X, sess, xform=xform)
            multiple_predictions.append(predictions)
        return np.mean(multiple_predictions, axis=0)


class TenCropPredictor(PredictSessionMixin):
    def __init__(self, model, cnf, weights_from, prediction_iterator, im_size, crop_size, output_layer='predictions'):
        self.crop_size = crop_size
        self.im_size = im_size
        self.cnf = cnf
        self.prediction_iterator = prediction_iterator
        self.predictor = OneCropPredictor(model, cnf, weights_from, prediction_iterator, output_layer)
        super(TenCropPredictor, self).__init__(weights_from)

    def _build_model(self):
        self.predictor._build_model()

    def _real_predict(self, X, sess):
        crop_size = np.array(self.crop_size)
        im_size = np.array(self.im_size)
        bboxs = util.get_bbox_10crop(crop_size, im_size)
        multiple_predictions = []
        for i, bbox in enumerate(bboxs, start=1):
            print('Crop-deterministic iteration: %d' % i)
            predictions = self.predictor._real_predict(X, sess, crop_bbox=bbox)
            multiple_predictions.append(predictions)
        return np.mean(multiple_predictions, axis=0)


class EnsemblePredictor(object):
    def __init__(self, predictors):
        self.predictors = predictors

    def predict(self, X):
        multiple_predictions = []
        for p in self.predictors:
            print('Ensembler - running predictions using: %s' % p)
            predictions = p.predict(X)
            multiple_predictions.append(predictions)
            # Todo: introduce voting policies other than the arithmetic mean below
            # return np.mean(multiple_predictions, axis=0)
        return gmean(multiple_predictions, axis=0)

    def predict_with_voting(self, X, score_to_classes, vote_combiner):
        votes = []
        for i, p in enumerate(self.predictors):
            print('Ensembler - running predictions using: %s' % p)
            predictions = p.predict(X)
            votes.append(score_to_classes[i](predictions))
        return vote_combiner(votes)
