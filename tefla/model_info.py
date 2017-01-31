import click
import tefla.utils.util as util


@click.command()
@click.option('--model', default=None, show_default=True,
              help='Relative path to model.')
def model_info(model):
    util.check_required_program_args([model])
    model_def = util.load_module(model)
    model = model_def.model
    end_points = model(False, None)
    util.show_layer_shapes(end_points)
    util.show_vars()


if __name__ == '__main__':
    model_info()
