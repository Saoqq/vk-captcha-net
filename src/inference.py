import json
import os
from pathlib import Path
import numpy as np
import requests
import tensorflow as tf
from tensorflow import keras
from keras import layers

from model import decode_ctc

# Path to the images directory
data_dir = Path("../train/")

# Get list of all the images
images = sorted(list(map(str, list(data_dir.glob("*.jpg")))))
labels = [img.split(os.path.sep)[-1].split(".jpg")[0] for img in images]
characters = set(char for label in labels for char in label)
characters = sorted(list(characters))
max_length = max([len(label) for label in labels])

img_height = 64
img_width = 128

char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)

# Mapping integers back to original characters
num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)


def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.transpose(img, perm=[1, 0, 2])
    return img


def decode_batch_predictions(pred):
    pred = np.array(pred)
    input_len = np.ones(pred.shape[0]) * (pred.shape[1])
    # Use greedy search. For complex tasks, you can use beam search
    results = decode_ctc(max_length, input_len, pred)

    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8").replace("[UNK]", "")
        output_text.append(res)
    return output_text


model = keras.models.load_model('../.data/models/vk_captcha_net/1')
pred = model.predict(tf.stack([load_image(images[0])]))

# Example for tf serving
# data = {"inputs": tf.stack([load_image(images[0])]]).numpy().tolist()}
# pred = requests.post('http://localhost:8501/v1/models/vk_captcha_net/versions/1:predict', data=json.dumps(data))
# pred = pred.json()['outputs']

pred = tf.strings.reduce_join(num_to_char(pred)).numpy().decode("utf-8").replace("[UNK]", "")

