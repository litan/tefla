import os

import click
import numpy as np

from tefla.core.iter_ops import create_prediction_iter, convert_preprocessor
from tefla.core.prediction import QuasiCropPredictor, TenCropPredictor, OneCropPredictor
from tefla.da import data
from tefla.utils import util


@click.command()
@click.option('--model', help='Relative path to model.')
@click.option('--output_layer', default='predictions', show_default=True,
              help='Layer from which to extract features.')
@click.option('--training_cnf', help='Relative path to training config file.')
@click.option('--predict_dir', help='Directory with test images')
@click.option('--weights_from', help='Path to initial weights file.')
@click.option('--tag', default='results', help='Name of the dataset')
@click.option('--convert', is_flag=True,
              help='Convert/preprocess files before prediction.')
@click.option('--image_size', default=256, show_default=True,
              help='Image size for conversion.')
@click.option('--sync', is_flag=True,
              help='Do all processing on the calling thread.')
@click.option('--predict_type', default='quasi', show_default=True,
              help='Specify predict type: quasi, 1_crop or 10_crop')
def predict_command(model, output_layer, training_cnf, predict_dir, weights_from, tag, convert, image_size, sync,
            predict_type):
    util.check_required_program_args([model, training_cnf, predict_dir, weights_from])
    model_def = util.load_module(model)
    model = model_def.model
    cnf = util.load_module(training_cnf).cnf
    weights_from = str(weights_from)
    images = data.get_image_files(predict_dir)
    predictions = predict(model, model_def, output_layer, cnf, weights_from, convert, images, image_size, sync,
                          predict_type)

    prediction_results_dir = os.path.abspath(os.path.join(predict_dir, '..', 'predictions', tag))
    if not os.path.exists(prediction_results_dir):
        os.makedirs(prediction_results_dir)

    if output_layer == 'predictions':
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
    else:
        # feature extraction
        features_file = os.path.join(prediction_results_dir, 'features.npy')
        np.save(features_file, predictions)
        print('Features from layer: %s saved to: %s' % (output_layer, features_file))

def predict(model, model_def, output_layer, cnf, weights_from, convert,images, image_size, sync, predict_type):
    preprocessor = convert_preprocessor(image_size) if convert else None
    prediction_iterator = create_prediction_iter(cnf, model_def.crop_size, preprocessor, sync)

    if predict_type == 'quasi':
        predictor = QuasiCropPredictor(model, cnf, weights_from, prediction_iterator, 20, output_layer)
    elif predict_type == '1_crop':
        predictor = OneCropPredictor(model, cnf, weights_from, prediction_iterator, output_layer)
    elif predict_type == '10_crop':
        predictor = TenCropPredictor(model, cnf, weights_from, prediction_iterator, model_def.crop_size[0],
                                     model_def.image_size[0], output_layer)
    else:
        raise ValueError('Unknown predict_type: %s' % predict_type)
    predictions = predictor.predict(images)

    return predictions

if __name__ == '__main__':
    predict_command()
