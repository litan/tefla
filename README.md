# Tefla
Tefla is a deep learning mini-framework that sits on top of Tensorflow. Tefla's primary goal is to enable simple, stable, end-to-end deep learning. This means that Tefla supports:
* Data setup
 * [Batch preprocessing][1] and [data layout][12].
* Training
 * A [model definition DSL][2].
 * A [training config DSL][3].
 * [Data loading][4] with [data-augmentation][5] and rebalancing.
 * [Training][6] with support for visualization, logging, custom metrics, and most importantly - resumption of training from an earlier epoch with a new learning rate.
 * Pluggable [learning rate decay policies][7].
 * Stability and solidity - which translates to days and weeks of training without memory blowup and epoch time degradations.
* Tensorboard visualization of epoch metrics, augmented images, model graphs, and layer activations, weights and gradients.
* [Prediction][8] (with ensembling via mean score or voting).
* [Metrics][9] on prediction outputs.
* First class support for transfer learning and fine-tuning based on vgg16, resnet50, resnet101, and resnet152.
* Serving of models via a REST API (*coming soon*).

Tefla contains [command line scripts][10] to do batch preprocessing, training, prediction, and metrics, thus supporting a simple yet powerful deep learning workflow.

Documentation is coming soon. For now, the [mnist example(s)][11] can help you to get started.

Tefla is very much a work in progress. Contributions are welcome!

An interesting fork of tefla is available here: www.github.com/n3011/tefla. Both projects are evolving independently, with a cross-pollination of ideas.

[1]: https://github.com/litan/tefla/blob/master/tefla/convert.py
[2]: https://github.com/litan/tefla/blob/master/examples/mnist/mnist_model.py
[3]: https://github.com/litan/tefla/blob/master/examples/mnist/mnist_cnf.py
[4]: https://github.com/litan/tefla/blob/master/tefla/da/iterator.py
[5]: https://github.com/litan/tefla/blob/master/tefla/da/data.py
[6]: https://github.com/litan/tefla/blob/master/tefla/core/training.py
[7]: https://github.com/litan/tefla/blob/master/tefla/core/lr_policy.py
[8]: https://github.com/litan/tefla/blob/master/tefla/core/prediction.py
[9]: https://github.com/litan/tefla/blob/master/tefla/metrics.py
[10]: https://github.com/litan/tefla/blob/master/tefla/
[11]: https://github.com/litan/tefla/tree/master/examples/mnist
[12]: https://github.com/litan/tefla/blob/master/tefla/core/dir_dataset.py
