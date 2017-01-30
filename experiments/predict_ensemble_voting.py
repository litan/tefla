import os

import click
import numpy as np
from scipy.stats import mode

from tefla.core.iter_ops import create_prediction_iter, convert_preprocessor
from tefla.core.prediction import QuasiCropPredictor, OneCropPredictor, EnsemblePredictor
from tefla.da import data
from tefla.da.standardizer import NoOpStandardizer
from tefla.utils import util


@click.command()
@click.option('--model', default=None, show_default=True,
              help='Relative path to model.')
@click.option('--training_cnf', default=None, show_default=True,
              help='Relative path to training config file.')
@click.option('--predict_dir', help='Directory with Test Images')
@click.option('--weights_from', help='Path to initial weights file.')
@click.option('--dataset_name', default='dataset', help='Name of the dataset')
@click.option('--convert', is_flag=True,
              help='Convert/preprocess files before prediction.')
@click.option('--image_size', default=256, show_default=True,
              help='Image size for conversion.')
@click.option('--sync', is_flag=True,
              help='Do all processing on the calling thread.')
@click.option('--predict_type', default='quasi', show_default=True,
              help='Specify predict type: quasi, 1_crop or 10_crop')
def predict(model, training_cnf, predict_dir, weights_from, dataset_name, convert, image_size, sync,
            predict_type):
    images = data.get_image_files(predict_dir)

    # Form now, hard coded models, cnfs, and weights
    # Need to take these from program inputs or an ensembling config file

    print('Creating predictor 1')
    weights_from1 = 'weights.sa/model-epoch-97.ckpt'
    model1 = 'examples/mnist_model_sa.py'
    training_cnf1 = 'examples/mnist_cnf.py'
    model_def1 = util.load_module(model1)
    model1 = model_def1.model
    cnf1 = util.load_module(training_cnf1).cnf
    standardizer = cnf1.get('standardizer', NoOpStandardizer())
    preprocessor = convert_preprocessor(model_def1.image_size[0]) if convert else None
    prediction_iterator1 = create_prediction_iter(cnf1, standardizer, model_def1.crop_size, preprocessor, sync)
    # predictor1 = QuasiCropPredictor(model1, cnf1, weights_from1, prediction_iterator1, 20)
    predictor1 = OneCropPredictor(model1, cnf1, weights_from1, prediction_iterator1)

    print('Creating predictor 2')
    weights_from2 = 'weights.rv/model-epoch-31.ckpt'
    model2 = 'examples/mnist_model.py'
    training_cnf2 = 'examples/mnist_cnf.py'
    model_def2 = util.load_module(model2)
    model2 = model_def2.model
    cnf2 = util.load_module(training_cnf2).cnf
    standardizer = cnf2.get('standardizer', NoOpStandardizer())
    preprocessor = convert_preprocessor(model_def2.image_size[0]) if convert else None
    prediction_iterator2 = create_prediction_iter(cnf2, standardizer, model_def2.crop_size, preprocessor, sync)
    # predictor2 = QuasiCropPredictor(model2, cnf2, weights_from2, prediction_iterator2, 20)
    predictor2 = OneCropPredictor(model2, cnf2, weights_from2, prediction_iterator2)

    predictor = EnsemblePredictor([predictor1, predictor2])

    def softmax_result_to_vote(predictions):
        return predictions.argmax(axis=1)

    def vote_combiner(votes):
        return mode(votes, axis=0)[0].reshape(-1)

    class_predictions = predictor.predict_with_voting(
        images,
        [softmax_result_to_vote, softmax_result_to_vote],
        vote_combiner
    )

    if not os.path.exists(os.path.join(predict_dir, '..', 'results')):
        os.mkdir(os.path.join(predict_dir, '..', 'results'))
    if not os.path.exists(os.path.join(predict_dir, '..', 'results', dataset_name)):
        os.mkdir(os.path.join(predict_dir, '..', 'results', dataset_name))

    names = data.get_names(images)
    image_class_predictions = np.column_stack([names, class_predictions])
    title = np.array(['image', 'label'])
    image_class_predictions = np.vstack([title, image_class_predictions])
    prediction_class_file = os.path.abspath(
        os.path.join(predict_dir, '..', 'results', dataset_name, 'predictions_class.csv'))
    np.savetxt(prediction_class_file, image_class_predictions, delimiter=",", fmt="%s")
    print('Class predictions saved to: %s' % prediction_class_file)


if __name__ == '__main__':
    predict()
