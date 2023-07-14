import numpy as np
import imageio.v2  as imageio
from tensorflow.keras import Sequential as Seq
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.datasets import mnist
from sklearn.metrics import classification_report

def _init_img_data():
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  x_train = x_train / 255
  x_test = x_test / 255

  # Convert Y to one-hot encoding
  y_train_t = np.zeros((y_train.shape[0], y_train.max() + 1))
  y_train_t[np.arange(y_train.shape[0]), y_train] = 1

  y_test_t = np.zeros((y_test.shape[0], y_test.max() + 1))
  y_test_t[np.arange(y_test.shape[0]), y_test] = 1 ## y_test_t[row(0->N), y_test[0->N]] = 1

  return x_test, x_train, y_train_t, y_test_t, y_train, y_test


def flatten_bin_img(X):
    # Flattens the input binary image from (n, l, w) to (n, l*w)
    return X.reshape(X.shape[0], -1)

# Custom Softmax implementation
def custom_impl():
  from softmax import Layer, LINEAR, \
    initialize_parameters, \
    L_model_f_prop, \
    compute_crossentropy_loss, \
    L_model_b_prop, \
    update_params, \
    predict
  '''
  higher accuracy requires additional cycles or adjustments
  to the neurel network architecture. This can be easily done
  by adding more Layer objects to the parameter list below.
      accuracy  0.87     10000
  '''
  x_test, x_train, y_train, y_test, y_train_orig, y_test_orig = _init_img_data()
  x_train = flatten_bin_img(x_train).T
  x_test = flatten_bin_img(x_test).T

  param = initialize_parameters([
    Layer(
      layer_name="Input",
      layer_dims=x_test.shape[0],
      activation_function=None
    ),      
    Layer(
      layer_name="L1",
      layer_dims=150
    ),
    Layer(
      layer_name="Output",
      layer_dims=10,
      activation_function=LINEAR
    )
  ])
  # Note: MNIST runs over 60_000 images which significantly impacts performance.
  # Switching to train on the test dataset is a good idea to quickly iterate. For 
  # most accurate prediction results use the train dataset.
  grads = None
  for ii in range(200):
    R, ca = L_model_f_prop(x_train, param)
    if ii % 50 == 0:
      print(f"Cost: {compute_crossentropy_loss(R, y_train)}")
    grads = L_model_b_prop(R, y_train, ca)
    param = update_params(param, grads, learning_rate=.5)
  res = predict(param, x_test)
  print(classification_report(y_test_orig, res))


## Tensorflow implementation for Softmax classification
def tf_impl():
  '''
      accuracy  0.97     10000
  '''
  x_test, x_train, y_train, y_test = _init_img_data()

  model = Seq([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu', name="L1"),
    Dense(10, activation='linear', name="L2")
  ])
  model.compile(
      loss=SparseCategoricalCrossentropy(from_logits=True)
  )
  model.fit(
      x_train,
      y_train,
      epochs=3
  )
  res =  np.argmax(model.predict(x_test), axis=1)
  print(classification_report(y_test, res))



## Run
custom_impl()
