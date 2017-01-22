This directory contains multiple mnist examples:
* mnist.py - is based on a simple model with fc layers. Run with `python mnist.py`
* mnist_conv.py - is based on a model with conv layers. Run with `python mnist_conv.py`

The remaining files help you play with mnist in a manner similar to what you would do to work 
with a real world dataset. Here's a brief description of the files:
* mnist_save* scripts - help you put the mnist data inside a directory, as per tefla conventions.
* mnist_model.py - contains a deep net model definition.
* mnist_cnf.py - is the training config file.

To train, run:
```
python -m tefla.train --model examples/mnist/mnist_model.py --training_cnf examples/mnist/mnist_cnf.py 
--data_dir /path/to/data
```

To do predictions and metrics, play with:
```
python -m tefla.predict --help
python -m tefla.metrics --help
```
 