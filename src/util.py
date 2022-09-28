import os
from pathlib import Path
from keras import layers

img_height = 64
img_width = 128

batch_size = 16

data_dir = Path("../train/")

images = sorted(list(map(str, list(data_dir.glob("*.jpg")))))
labels = [img.split(os.path.sep)[-1].split(".jpg")[0] for img in images]

characters = set(char for label in labels for char in label)
characters = sorted(list(characters))

max_length = max([len(label) for label in labels])

char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)

# Mapping integers back to original characters
num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)
