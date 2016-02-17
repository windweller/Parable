# parable

A highly customizable lightweight library on Neural Network.

Unlike its counterparts (i.e., Lasagne, Keras), Parable builds a "shallow" (or centralized) network structure. Most Neural Network libraries store weights and manage weights inside various `Layer` classes (such as `Conv2DLayer()` or `DenseLayer()`). Parable however, stores and manages weights and biases inside the `Network` class, and distribute those into various layers when expression are being evaluated.