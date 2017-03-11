"""
This is a script for doing predictions based on input features as opposed to input images.
The input features are meant to be bottleneck features extracted from a well known net.
"""
import os

import click
import numpy as np

from tefla.core.prediction import InputFeaturesPredictor
from tefla.da import data
from tefla.utils import util


@click.command()
@click.option('--model', help='Relative path to model.')
@click.option('--training_cnf', help='Relative path to training config file.')
@click.option('--features_file', help='File with input features')
@click.option('--images_dir', help='Dir with images that are the source of the input features')
@click.option('--weights_from', help='Path to initial weights file.')
@click.option('--tag', default='results', help='Name of the dataset')
@click.option('--sync', is_flag=True,
              help='Do all processing on the calling thread.')
def predict_command(model, training_cnf, features_file, images_dir, weights_from, tag, sync, ):
    util.check_required_program_args([model, training_cnf, features_file, images_dir, weights_from])
    model_def = util.load_module(model)
    model = model_def.model
    cnf = util.load_module(training_cnf).cnf
    weights_from = str(weights_from)
    image_features = np.load(features_file)
    images = data.get_image_files(images_dir)
    predictions = predict_withf(model, cnf, weights_from, image_features)
    predict_dir = os.path.dirname(features_file)
    prediction_results_dir = os.path.abspath(os.path.join(predict_dir, 'predictions', tag))
    if not os.path.exists(prediction_results_dir):
        os.makedirs(prediction_results_dir)

    names = data.get_names(images)
    image_prediction_probs = np.column_stack([names, predictions])
    headers = ['score%d' % (i + 1) for i in range(predictions.shape[1])]
    title = np.array(['image'] + headers)
    image_prediction_probs = np.vstack([title, image_prediction_probs])
    prediction_probs_file = os.path.join(prediction_results_dir, 'predictions.csv')
    np.savetxt(prediction_probs_file, image_prediction_probs, delimiter=",", fmt="%s")
    print('Predictions saved to: %s' % prediction_probs_file)
    if cnf['classification']:
        class_predictions = np.argmax(predictions, axis=1)
        image_class_predictions = np.column_stack([names, class_predictions])
        title = np.array(['image', 'label'])
        image_class_predictions = np.vstack([title, image_class_predictions])
        prediction_class_file = os.path.join(prediction_results_dir, 'predictions_class.csv')
        np.savetxt(prediction_class_file, image_class_predictions, delimiter=",", fmt="%s")
        print('Class predictions saved to: %s' % prediction_class_file)

def predict_withf(model, cnf, weights_from,image_features):
    predictor = InputFeaturesPredictor(model, cnf, weights_from)
    predictions = predictor.predict(image_features)
    return predictions


if __name__ == '__main__':
    predict_command()
