import numpy as np
from tensorflow import keras
import tensorflow as tf

model = keras.models.load_model('../.data/training/vk_captcha_net')

x = model.get_layer(name="cropping1d").output

# adding that layer frees you from running CTC decoding on client
# FIXME: only batch of single item will be acceptable as input with such layer
x = tf.keras.layers.Lambda(lambda y_pred: keras.backend.ctc_decode(
    y_pred,
    input_length=np.ones(1) * 30,
    greedy=True)[0][0][:, :7], name="decode_ctc")(x)

prediction_model = keras.models.Model(model.get_layer(name="image").input, x)

prediction_model.save("../.data/models/vk_captcha_net/1")
