import click
import numpy as np

from imagenet_classes import class_names
from tefla.core.iter_ops import create_prediction_iter
from tefla.core.prediction import QuasiCropPredictor, TenCropPredictor, OneCropPredictor
from tefla.da import data
from tefla.utils import util


@click.command()
@click.option('--model', default=None, show_default=True,
              help='Relative path to model.')
@click.option('--training_cnf', default=None, show_default=True,
              help='Relative path to training config file.')
@click.option('--predict_dir', help='Directory with Test Images')
@click.option('--weights_from', help='Path to initial weights file.')
@click.option('--predict_type', default='1_crop', show_default=True,
              help='Specify predict type: quasi, 1_crop or 10_crop')
def predict(model, training_cnf, predict_dir, weights_from, predict_type):
    model_def = util.load_module(model)
    model = model_def.model
    cnf = util.load_module(training_cnf).cnf
    weights_from = str(weights_from)
    images = data.get_image_files(predict_dir)

    preprocessor = None
    prediction_iterator = create_prediction_iter(cnf, model_def.crop_size, preprocessor, False)

    if predict_type == 'quasi':
        predictor = QuasiCropPredictor(model, cnf, weights_from, prediction_iterator, 20)
    elif predict_type == '1_crop':
        predictor = OneCropPredictor(model, cnf, weights_from, prediction_iterator)
    elif predict_type == '10_crop':
        predictor = TenCropPredictor(model, cnf, weights_from, prediction_iterator, model_def.crop_size[0],
                                     model_def.image_size[0])
    else:
        raise ValueError('Unknown predict_type: %s' % predict_type)
    predictions = predictor.predict(images)
    predictions = predictions.reshape(-1, 1000)

    names = data.get_names(images)
    for i, name in enumerate(names):
        print("---Predictions for %s:" % name)
        preds = (np.argsort(predictions[i])[::-1])[0:5]
        for p in preds:
            print(class_names[p], predictions[i][p])


if __name__ == '__main__':
    predict()
