import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from functools import partial
from albumentations import Compose, OneOf, MotionBlur, MedianBlur, Blur, Downscale, RGBShift, \
    GaussNoise, ShiftScaleRotate, OpticalDistortion, GridDistortion, Sharpen, Emboss, ImageCompression
from pathlib import Path
from tensorflow import keras
from keras import layers
from model import build_model, decode_ctc

# Path to the images directory
data_dir = Path("../train/")

# Get list of all the images
images = sorted(list(map(str, list(data_dir.glob("*.jpg")))))
labels = [img.split(os.path.sep)[-1].split(".jpg")[0] for img in images]
characters = set(char for label in labels for char in label)
characters = sorted(list(characters))

# Maximum length of any captcha in the dataset
max_length = max([len(label) for label in labels])

labels = [item.ljust(max_length) for item in labels]

print("Number of images found: ", len(images))
print("Number of labels found: ", len(labels))
print("Number of unique characters: ", len(characters))
print("Characters present: ", characters)

# Batch size for training and validation
batch_size = 16
# Desired image dimensions
# FIXME: probably won't work on other sizes, need to adjust model construction
img_width = 128
img_height = 64

# Mapping characters to integers
char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)

# Mapping integers back to original characters
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)


def split_data(images, labels, train_size=0.9, shuffle=True):
    # 1. Get the total size of the dataset
    size = len(images)
    # 2. Make an indices array and shuffle it, if required
    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)
    # 3. Get the size of training samples
    train_samples = int(size * train_size)
    # 4. Split data into training and validation sets
    x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
    x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
    return x_train, x_valid, y_train, y_valid


# Splitting data into training and validation sets
# FIXME: use tf.keras.utils.split_dataset instead?
x_train, x_valid, y_train, y_valid = split_data(np.array(images), np.array(labels))


def prepare_single_sample(img_path, label):
    # 1. Read image
    img = tf.io.read_file(img_path)
    # 2. Decode and convert to grayscale
    img = tf.io.decode_jpeg(img)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize to the desired size
    img = tf.image.resize(img, [img_height, img_width])
    # 5. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])
    # 6. Map the characters in label to numbers
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    # 7. Return a dict as our model is expecting two inputs
    # And also returning y_true so categorical_acc will be calculated automatically
    return {"image": img, "label": label}, label


aug = Compose([
    # add one of basic image quality reduction augmentations
    OneOf([
        OneOf([
            MotionBlur(blur_limit=(3, 3), p=0.1),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=(3, 3), p=0.15),
            ImageCompression(quality_lower=30, quality_upper=80, p=0.1),
            Downscale(p=0.2, scale_min=0.5, scale_max=0.75),
        ], p=0.2),

    ], p=0.1),
    # add color augmentation
    RGBShift(r_shift_limit=(0, 0.2), g_shift_limit=(0, 0.2), b_shift_limit=(0, 0.2), p=0.4),
    # add one of basic image noise applying augmentations
    OneOf([
        GaussNoise(p=0.2, var_limit=(0.0001, 0.001)),
        Sharpen(p=0.2),
        Emboss(p=0.1),
    ], p=0.4),
    # add scale augmentation
    ShiftScaleRotate(rotate_limit=5, shift_limit=0.0525, scale_limit=(-0.15, 0.3), p=0.4),
    # add distortion and perspective augmentations
    OneOf([
        OpticalDistortion(p=0.5),
        GridDistortion(p=0.2),
    ], p=0.5),
], p=0.9)


# wrapping augmentations fun to be compatible with tf dataset pipeline
def process_data(data, target):
    def aug_fn(image):
        data = {"image": image}
        aug_data = aug(**data)
        aug_img = aug_data["image"]
        aug_img = tf.cast(aug_img, tf.float32)
        return aug_img

    aug_img = tf.numpy_function(func=aug_fn, inp=[data["image"]], Tout=tf.float32)
    return {"image": aug_img, "label": target}, target


# FIXME: decode first into uint8 then augment and then cast to float,
#  didn't do that way as wanted to keep same load functions for train and validate datasets
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = (
    train_dataset.map(prepare_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .map(partial(process_data), num_parallel_calls=tf.data.AUTOTUNE)
    # making batch always full, required for decoding at CTC loss layer
    .batch(batch_size, drop_remainder=True)
    .repeat(10)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
validation_dataset = (
    validation_dataset.map(prepare_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size, drop_remainder=True)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

_, ax = plt.subplots(4, 4, figsize=(10, 5))
for batch, _ in train_dataset.take(1):
    images = batch["image"]
    labels = batch["label"]
    for i in range(16):
        img = (images[i] * 255).numpy().astype("uint8")
        label = tf.strings.reduce_join(num_to_char(labels[i])).numpy().decode("utf-8")
        # transpose back to normal image for vis
        ax[i // 4, i % 4].imshow(np.transpose(img, (1, 0, 2)))
        ax[i // 4, i % 4].set_title(label)
        ax[i // 4, i % 4].axis("off")
plt.show()

# Get the model
model = build_model(img_width, img_height, batch_size, max_length, len(char_to_num.get_vocabulary()))
model.summary()

epochs = 100
early_stopping_patience = 10
# Add early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    callbacks=[early_stopping],
)
model.save("../.data/training/vk_captcha_net")
# Get the prediction model by extracting layers till the output layer
prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output
)
prediction_model.summary()


# Decode predictions after inference, no CTC layer, output becomes non-decoded
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * (pred.shape[1])
    # Use greedy search. For complex tasks, you can use beam search
    results = decode_ctc(max_length, input_len, pred)

    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8").replace("[UNK]", "")
        output_text.append(res)
    return output_text


#  Check results on validation dataset
for batch, _ in validation_dataset.take(1):
    batch_images = batch["image"]
    batch_labels = batch["label"]

    preds = prediction_model.predict(batch_images)
    pred_texts = decode_batch_predictions(preds)

    orig_texts = []
    for label in batch_labels:
        label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
        orig_texts.append(label)

    _, ax = plt.subplots(4, 4, figsize=(15, 5))
    for i in range(16):
        img = (batch_images[i] * 255).numpy().astype("uint8")
        title = f"Prediction: {pred_texts[i]}"
        ax[i // 4, i % 4].imshow(np.transpose(img, (1, 0, 2)))
        ax[i // 4, i % 4].set_title(title)
        ax[i // 4, i % 4].axis("off")
plt.show()
