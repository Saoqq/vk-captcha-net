import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow import keras
from keras import layers


def decode_ctc(max_length, input_length, y_pred):
    return keras.backend.ctc_decode(y_pred, input_length)[0][0][:, :max_length]


class CTCLayer(layers.Layer):
    def __init__(self, batch_size, max_length, name=None):
        super().__init__(name=name)
        self.batch_size = batch_size
        self.max_length = max_length
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`
        batch_len = tf.cast(tf.shape(y_pred)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = tf.math.count_nonzero(y_true, 1, keepdims=True)

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred


def CTCDecoder(max_length, name='ctc_decoder'):
    def decoder(y_pred):
        batch_size = tf.cast(tf.shape(y_pred)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        decoded = keras.backend.ctc_decode(
            y_pred, input_length * tf.ones(shape=batch_size, dtype="int64")
        )[0][0][:, :max_length]
        return decoded

    return tf.keras.layers.Lambda(decoder, name=name)


def build_model(img_width, img_height, batch_size, max_length, num_classes, training=True):
    def conv_module(layer, K, kX, kY, stride=(1, 1), chan_dim=-1, padding='same'):
        '''Inception Conv Layer
        Args:
            layer (tf.keras.Layer): input layer.
            K (int): number of filters,
            kX (int): kernel size at X dimension.
            kY (int): kernel size at Y dimension.
            stride (tuple, default is (1, 1)): strides for convolution
            chan_dim (int, default is -1): channels format, either last (-1) or first.
            padding (str, default='same'): padding for convolution
            trainable (bool, default is True): if False then BatchNormalization
                won't update weights
        Returns:
            new_layer (tf.keras.Layer): new Layer after Conv+BN+Act
        '''
        layer = layers.Conv2D(K, (kX, kY), strides=stride, padding=padding,
                              kernel_initializer='he_normal')(layer)
        layer = layers.BatchNormalization(axis=chan_dim)(layer)
        layer = layers.Activation('relu')(layer)

        return layer

    def conv_module_fact(layer, K=64, kX=3, kY=3, stride=(1, 1), chan_dim=-1, padding='same'):
        '''Factorized Inception Conv Block with fewer parameters.
        Args:
            layer (tf.keras.Layer): input layer.
            K (int): number of filters,
            kX (int): kernel size at X dimension.
            kY (int): kernel size at Y dimension.
            stride (tuple, default is (1, 1)): strides for convolution
            chan_dim (int, default is -1): channels format, either last (-1) or first.
            padding (str, default='same'): padding for convolution
            trainable (bool, default is True): if False, then BatchNormalization
                won't update weights
        Returns:
            new_layer (tf.keras.Layer): new Layer after Conv+BN+Act
        '''
        layer = layers.Conv2D(K, (1, kY), strides=(1, stride[-1]), padding=padding,
                              kernel_initializer='he_normal')(layer)
        layer = layers.Conv2D(K, (kX, 1), strides=(stride[0], 1), padding=padding,
                              kernel_initializer='he_normal')(layer)
        layer = layers.BatchNormalization(axis=chan_dim)(layer)
        layer = layers.Activation('relu')(layer)

        return layer

    def inceptionv1_module(layer, numK1x1, numK3x3, chan_dim=-1):
        '''Factorized InceptionV2 Conv Module.
        Args:
            layer (tf.keras.Layer): input layer.
            numK1x1 (int): number of filters in 1x1 convolution,
            numK3x3 (int): number of filters in 3x3 factorized convolution.
            chan_dim (int, default is -1): channels format, either last (-1) or first.
            trainable (bool, default is True): if False, then BatchNormalization
                won't update weights
        Returns:
            new_layer (tf.keras.Layer): new Layer after Factorized InceptionV2 Conv Module
        '''
        conv_1x1 = conv_module(layer, numK1x1, 1, 1, (1, 1),
                               chan_dim)
        conv_3x3 = conv_module_fact(layer, numK3x3, 3, 3, (1, 1),
                                    chan_dim)
        layer = layers.concatenate([conv_1x1, conv_3x3], axis=chan_dim)

        return layer

    # Inputs to the model
    input_img = layers.Input(shape=(img_width, img_height, 3), name="image", dtype="float32")
    # Needed for CTC loss calculation
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    x = inceptionv1_module(input_img, 32, 64)

    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    x = inceptionv1_module(x, 64, 128)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    x = inceptionv1_module(x, 128, 256)
    x = inceptionv1_module(x, 128, 256)
    x = layers.MaxPooling2D(pool_size=(1, 2), name='pool3')(x)

    x = inceptionv1_module(x, 256, 512)
    x = layers.MaxPooling2D(pool_size=(1, 2), name='pool4')(x)

    x = layers.Conv2D(512, (2, 2), strides=(1, 1), padding='same', kernel_initializer='he_normal',
                      name='con7')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    new_shape = ((img_width // 4), (img_height // 2) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    # not active during inference
    x = layers.Dropout(0.05)(x)
    x = layers.Dense(64, activation="relu", kernel_initializer='he_normal', name="dense1")(x)

    gru_1 = layers.GRU(256, return_sequences=True, kernel_initializer='he_normal', name='gru1')(x)
    gru_1b = layers.GRU(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(x)
    reversed_gru_1b = layers.Lambda(lambda inputTensor: tf.reverse(inputTensor, axis=[1]))(gru_1b)

    gru1_merged = layers.add([gru_1, reversed_gru_1b])
    gru1_merged = layers.BatchNormalization()(gru1_merged)

    gru_2 = layers.GRU(256, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = layers.GRU(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(
        gru1_merged)
    reversed_gru_2b = layers.Lambda(lambda inputTensor: tf.reverse(inputTensor, axis=[1]))(gru_2b)

    gru2_merged = layers.concatenate([gru_2, reversed_gru_2b])
    gru2_merged = layers.BatchNormalization()(gru2_merged)

    x = layers.Dense(num_classes + 1, activation="softmax", name="dense2")(gru2_merged)

    # Next comment on model behavior may be incorrect, that's how I understand it:
    # Removing 1 and -1 time features as usually captcha images don't have anything at the start and in the end of image
    # That should result in better handling of short captcha: 3-5 symbols centered in the middle
    # BUT negatively affects on long captcha: 5-7 symbols spread across all image
    x = layers.Cropping1D(cropping=1, name="cropping")(x)

    # Add CTC layer for calculating CTC loss at each step
    # This layer removed for inference
    if training:
        x = CTCLayer(batch_size, max_length, name="ctc_loss")(labels, x)
        output = CTCDecoder(max_length, name="ctc_decoder")(x)
    else:
        output = CTCDecoder(max_length, name="ctc_decoder")(x)

    # Define the model
    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="vk_captcha")
    # Optimizer
    opt = keras.optimizers.Adam()
    # Compile the model and return
    model.compile(optimizer=opt,
                  metrics=['categorical_accuracy']
                  )
    return model


def cast_to_inference(model):
    x = model.get_layer(name="cropping").output
    x = model.get_layer(name="ctc_decoder")(x)
    prediction_model = keras.models.Model(model.get_layer(name="image").input, x)
    return prediction_model
