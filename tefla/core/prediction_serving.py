from __future__ import division, print_function, absolute_import

import time

import numpy as np
import tensorflow as tf
from scipy.stats.mstats import gmean

from tefla.da import tta
from tefla.da.iterator import BatchIterator
from tefla.utils import util


class ServingSessionMixin(object):
    def __init__(self, weights_from):
        self.weights_from = weights_from
        self.inference_graph = tf.Graph()
        with self.inference_graph.as_default():
            self._build_model()
            saver = tf.train.Saver()
            self.inference_session = tf.Session()
            print('Loading weights from: %s' % self.weights_from)
            util.load_variables(self.inference_session, saver, self.weights_from)

    def predict(self, X, **kwargs):
        return self._real_predict(X, self.inference_session, **kwargs)

    def _real_predict(self, X, sess, **kwargs):
        pass

    def _build_model(self):
        pass


class OneCropPredictor(ServingSessionMixin):
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
        print('took %6.2f seconds' % (time.time() - tic))
        return data_predictions


class InputFeaturesPredictor(ServingSessionMixin):
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

    def _real_predict(self, X, sess, **kwargs):
        tic = time.time()
        print('Making %d predictions' % len(X))
        data_predictions = []
        for X, y in self.prediction_iterator(X):
            predictions_e = sess.run(self.predictions, feed_dict={self.inputs: X})
            data_predictions.append(predictions_e)
        data_predictions = np.vstack(data_predictions)
        print('took %6.2f seconds' % (time.time() - tic))
        return data_predictions


class QuasiCropPredictor():
    def __init__(self, model, cnf, weights_from, prediction_iterator, number_of_transforms, output_layer='predictions'):
        self.number_of_transforms = number_of_transforms
        self.cnf = cnf
        self.prediction_iterator = prediction_iterator
        self.predictor = OneCropPredictor(model, cnf, weights_from, prediction_iterator, output_layer)

    def predict(self, X):
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
            predictions = self.predictor.predict(X, xform=xform)
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
