在继承 tf.keras.model() 的时候，我们可以选择性地设置一个参数 training (boolean)， 区别提现在 training模式还是inference模式

- During training, dropout will randomly drop out units and correspondingly scale up activations of the remaining units.
- During inference, it does nothing (since you usually don't want the randomness of dropping out units here).





  **About setting `layer.trainable = False` on a `BatchNormalization layer:**
  The meaning of setting `layer.trainable = False` is to freeze the layer,
  i.e. its internal state will not change during training:
  its trainable weights will not be updated
  during `fit()` or `train_on_batch()`, and its state updates will not be run.

  Usually, this does not necessarily mean that the layer is run in inference
  mode (which is normally controlled by the `training` argument that can
  be passed when calling a layer). "Frozen state" and "inference mode"
  are two separate concepts.

  However, in the case of the `BatchNormalization` layer, **setting
  `trainable = False` on the layer means that the layer will be
  subsequently run in inference mode** (meaning that it will use
  the moving mean and the moving variance to normalize the current batch,
  rather than using the mean and variance of the current batch).

  This behavior has been introduced in TensorFlow 2.0, in order
  to enable `layer.trainable = False` to produce the most commonly
  expected behavior in the convnet fine-tuning use case.
  Note that:

- This behavior only occurs as of TensorFlow 2.0. In 1.*,
setting `layer.trainable = False` would freeze the layer but would
not switch it to inference mode.
- Setting `trainable` on an model containing other layers will
recursively set the `trainable` value of all inner layers.
- If the value of the `trainable`
attribute is changed after calling `compile()` on a model,
the new value doesn't take effect for this model
until `compile()` is called again.
''')])

