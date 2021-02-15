"""docstring"""
import logging

from keras.layers import (
    Conv2D,
    Conv2DTranspose,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
    UpSampling2D,
)
from keras.layers.experimental.preprocessing import Resizing
from keras.layers.merge import concatenate
from keras.models import Model

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

file_handler = logging.FileHandler("Models.log")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


def make_model_rad_regulizer(shape_tensor=(124, 124, 1), feature_dim=(3,)):     #pylint: disable=R0915
    """Modello CAE tratto da Liu et al, ma modificato con dropout, maxpooling e upsampling.
    Permette inoltre la classificazione delle masse grazie a layer fc.
    type shape_tensor: array
    param shape_tensor: dimensione dell'immagine in ingresso

    type feature_dim: array
    param feature_dim: dimensione dell'array delle feature

    :returns: restituisce il modello
    :rtype: keras model

    """
    input_tensor = Input(shape=shape_tensor, name="tensor_input")
    logger.debug(f"dimensione input immagine:{shape_tensor}")   #pylint: disable=W1203
    input_vector = Input(shape=feature_dim)
    logger.debug(f"dimensione input feature:{feature_dim}")     #pylint: disable=W1203

    x = Conv2D(32, (5, 5), strides=2, padding="same", activation="relu")(input_tensor)
    x = Dropout(
        0.2,
    )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding="same")(x)
    x = Conv2D(64, (3, 3), strides=2, padding="same", activation="relu")(x)
    x = Dropout(
        0.2,
    )(x)
    x = Conv2D(
        128, (3, 3), strides=2, padding="same", activation="relu", name="last_conv"
    )(x)

    flat = Flatten()(x)
    flat = concatenate([flat, input_vector])
    den = Dense(16, activation="relu")(flat)
    # den= Dropout(.1,)(den)

    classification_output = Dense(
        2, activation="sigmoid", name="classification_output"
    )(den)

    x = Conv2DTranspose(64, (3, 3), strides=2, padding="same", activation="relu")(x)
    x = Dropout(
        0.2,
    )(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(32, (3, 3), strides=2, padding="same", activation="relu")(x)
    x = Conv2DTranspose(32, (3, 3), strides=2, padding="same", activation="relu")(x)
    decoder_out = Conv2D(
        1, (5, 5), padding="valid", activation="sigmoid", name="decoder_output"
    )(x)
    model = Model([input_tensor, input_vector], [decoder_out, classification_output])

    return model


def make_model_rad(shape_tensor=(124, 124, 1), feature_dim=(3,)):       #pylint: disable=R0915
    """Modello CAE tratto da Liu et al.
    Permette inoltre la classificazione delle masse grazie a layer fc.
    type shape_tensor: array
    param shape_tensor: dimensione dell'immagine in ingresso

    type feature_dim: array
    param feature_dim: dimensione dell'array delle feature

    :returns: restituisce il modello
    :rtype: keras model

    """
    input_tensor = Input(shape=shape_tensor, name="tensor_input")
    logger.debug(f"dimensione input immagine:{shape_tensor}")       #pylint: disable=W1203
    input_vector = Input(shape=feature_dim)
    logger.debug(f"dimensione input feature:{feature_dim}")     #pylint: disable=W1203

    x = Conv2D(32, (5, 5), strides=2, padding="same", activation="relu")(input_tensor)

    x = Conv2D(64, (3, 3), strides=2, padding="same", activation="relu")(x)

    x = Conv2D(
        128, (3, 3), strides=2, padding="same", activation="relu", name="last_conv"
    )(x)

    flat = Flatten()(x)
    flat = concatenate([flat, input_vector])
    den = Dense(16, activation="relu")(flat)

    classification_output = Dense(
        2, activation="sigmoid", name="classification_output"
    )(den)

    x = Conv2DTranspose(64, (3, 3), strides=2, padding="same", activation="relu")(x)
    x = Conv2DTranspose(32, (3, 3), strides=2, padding="same", activation="relu")(x)
    x = Conv2DTranspose(32, (3, 3), strides=2, padding="same", activation="relu")(x)
    decoder_out = Conv2D(
        1, (5, 5), padding="valid", activation="sigmoid", name="decoder_output"
    )(x)
    model = Model([input_tensor, input_vector], [decoder_out, classification_output])

    return model


def make_model_rad_unet(shape_tensor=(124, 124, 1), feature_dim=(3,)):      #pylint: disable=R0915
    """Modello UNET modificato con layer di resize.
    Permette inoltre la classificazione delle masse grazie a layer fc
    type shape_tensor: array
    param shape_tensor: dimensione dell'immagine in ingresso

    type feature_dim: array
    param feature_dim: dimensione dell'array delle feature

    :returns: restituisce il modello
    :rtype: keras model

    """
    input_tensor = Input(shape=shape_tensor, name="tensor_input")
    logger.debug(f"dimensione input immagine:{shape_tensor}")       #pylint: disable=W1203
    input_vector = Input(shape=feature_dim)
    logger.debug(f"dimensione input feature:{feature_dim}")     #pylint: disable=W1203

    c1 = Conv2D(
        16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(input_tensor)
    c1 = Dropout(0.2)(c1)
    c1 = Conv2D(
        16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(
        32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(
        32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Resizing(32, 32, interpolation="nearest")(p2)
    c3 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(p2)

    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Resizing(16, 16, interpolation="nearest")(p3)
    c4 = Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c4)

    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(
        256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(p4)

    c5 = Dropout(0.2)(c5)
    c5 = Conv2D(
        256,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="last_conv",
    )(c5)
    # fc layers

    flat = Flatten()(c5)
    flat = concatenate([flat, input_vector])
    den = Dense(16, activation="relu")(flat)

    classification_output = Dense(
        2, activation="sigmoid", name="classification_output"
    )(den)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c5)

    u6 = concatenate([u6, c4])
    c6 = Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c6)

    u7 = concatenate([u7, c3])
    c7 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c7)
    u8 = Resizing(62, 62, interpolation="nearest")(c2)

    u8 = concatenate([u8, c2])
    c8 = Conv2D(
        32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u8)
    c8 = Dropout(0.2)(c8)
    c8 = Conv2D(
        32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c8)

    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(
        16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u9)
    c9 = Dropout(0.2)(c9)
    c9 = Conv2D(
        16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c9)

    decoder_out = Conv2D(1, (1, 1), activation="sigmoid", name="decoder_output")(c9)

    model = Model([input_tensor, input_vector], [decoder_out, classification_output])
    return model


def make_model(shape_tensor=(124, 124, 1)):     #pylint: disable=R0915
    """Modello CAE tratto da Liu et al.
    Permette inoltre la classificazione delle masse grazie a layer fc.
    type shape_tensor: array
    param shape_tensor: dimensione dell'immagine in ingresso


    :returns: restituisce il modello
    :rtype: keras model

    """

    input_tensor = Input(shape=shape_tensor, name="tensor_input")
    logger.debug(f"dimensione input immagine:{shape_tensor}")       #pylint: disable=W1203

    x = Conv2D(32, (5, 5), strides=2, padding="same", activation="relu")(input_tensor)
    x = Conv2D(64, (3, 3), strides=2, padding="same", activation="relu")(x)
    x = Conv2D(
        128, (3, 3), strides=2, padding="same", activation="relu", name="last_conv"
    )(x)

    flat = Flatten()(x)
    den = Dense(16, activation="relu")(flat)
    classification_output = Dense(
        2, activation="sigmoid", name="classification_output"
    )(den)

    x = Conv2DTranspose(64, (3, 3), strides=2, padding="same", activation="relu")(x)
    x = Conv2DTranspose(32, (3, 3), strides=2, padding="same", activation="relu")(x)
    x = Conv2DTranspose(32, (3, 3), strides=2, padding="same", activation="relu")(x)
    decoder_out = Conv2D(
        1, (5, 5), padding="valid", activation="sigmoid", name="decoder_output"
    )(x)
    model = Model(input_tensor, [decoder_out, classification_output])

    return model


def make_model_regulizer(shape_tensor=(124, 124, 1)):       #pylint: disable=R0915
    """Modello CAE tratto da Liu et al, ma modificato con dropout, maxpooling e upsampling.
    Permette inoltre la classificazione delle masse grazie a layer fc.
    type shape_tensor: array
    param shape_tensor: dimensione dell'immagine in ingresso


    :returns: restituisce il modello
    :rtype: keras model

    """
    input_tensor = Input(shape=shape_tensor, name="tensor_input")
    logger.debug(f"dimensione input immagine:{shape_tensor}")       #pylint: disable=W1203
    x = Conv2D(32, (5, 5), strides=2, padding="same", activation="relu")(input_tensor)
    x = Dropout(
        0.2,
    )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding="same")(x)
    x = Conv2D(64, (3, 3), strides=2, padding="same", activation="relu")(x)
    x = Dropout(
        0.2,
    )(x)
    x = Conv2D(
        128, (3, 3), strides=2, padding="same", activation="relu", name="last_conv"
    )(x)

    flat = Flatten()(x)
    den = Dense(16, activation="relu")(flat)
    classification_output = Dense(
        2, activation="sigmoid", name="classification_output"
    )(den)

    x = Conv2DTranspose(64, (3, 3), strides=2, padding="same", activation="relu")(x)
    x = Dropout(
        0.2,
    )(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(32, (3, 3), strides=2, padding="same", activation="relu")(x)
    x = Conv2DTranspose(32, (3, 3), strides=2, padding="same", activation="relu")(x)
    decoder_out = Conv2D(
        1, (5, 5), padding="valid", activation="sigmoid", name="decoder_output"
    )(x)
    model = Model(input_tensor, [decoder_out, classification_output])

    return model


def make_model_unet(shape_tensor=(124, 124, 1)):        #pylint: disable=R0915
    """Modello Unet modificato con layer di resize.
    Permette inoltre la classificazione delle masse grazie a layer fc.
    type shape_tensor: array
    param shape_tensor: dimensione dell'immagine in ingresso


    :returns: restituisce il modello
    :rtype: keras model

    """
    input_tensor = Input(shape=shape_tensor, name="tensor_input")
    logger.debug(f"dimensione input immagine:{shape_tensor}")       #pylint: disable=W1203
    c1 = Conv2D(
        16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(input_tensor)
    c1 = Dropout(0.2)(c1)
    c1 = Conv2D(
        16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(
        32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(
        32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Resizing(32, 32, interpolation="nearest")(p2)
    c3 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(p2)

    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Resizing(16, 16, interpolation="nearest")(p3)
    c4 = Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c4)

    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(
        256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(p4)

    c5 = Dropout(0.2)(c5)
    c5 = Conv2D(
        256,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="last_conv",
    )(c5)
    # fc layers

    flat = Flatten()(c5)
    den = Dense(16, activation="relu")(flat)
    classification_output = Dense(
        2, activation="sigmoid", name="classification_output"
    )(den)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c5)

    u6 = concatenate([u6, c4])
    c6 = Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c6)

    u7 = concatenate([u7, c3])
    c7 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c7)
    u8 = Resizing(62, 62, interpolation="nearest")(c2)

    u8 = concatenate([u8, c2])
    c8 = Conv2D(
        32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u8)
    c8 = Dropout(0.2)(c8)
    c8 = Conv2D(
        32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c8)

    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(
        16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u9)
    c9 = Dropout(0.2)(c9)
    c9 = Conv2D(
        16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c9)

    decoder_out = Conv2D(1, (1, 1), activation="sigmoid", name="decoder_output")(c9)

    model = Model(input_tensor, [decoder_out, classification_output])
    return model


def make_model_rad_big_unet(shape_tensor=(4096, 3072, 1), feature_dim=(3,)):        #pylint: disable=R0915
    """Modello Unet.
    Permette inoltre la classificazione delle masse grazie a layer fc.
    type shape_tensor: array
    param shape_tensor: dimensione dell'immagine in ingresso

    type feature_dim: array
    param feature_dim: dimensione dell'array delle feature

    :returns: restituisce il modello
    :rtype: keras model

    """
    input_tensor = Input(shape=shape_tensor, name="tensor_input")
    logger.debug(f"dimensione input immagine:{shape_tensor}")       #pylint: disable=W1203
    input_vector = Input(shape=feature_dim)
    logger.debug(f"dimensione input feature:{feature_dim}")     #pylint: disable=W1203

    c1 = Conv2D(
        16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(input_tensor)
    c1 = Dropout(0.2)(c1)
    c1 = Conv2D(
        16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(
        32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(
        32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    c3 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(p2)

    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    c4 = Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c4)

    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(
        256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(p4)

    c5 = Dropout(0.2)(c5)
    c5 = Conv2D(
        256,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="last_conv",
    )(c5)
    # fc layers

    flat = Flatten()(c5)
    flat = concatenate([flat, input_vector])
    den = Dense(16, activation="relu")(flat)

    classification_output = Dense(
        2, activation="softmax", name="classification_output"
    )(den)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c5)

    u6 = concatenate([u6, c4])
    c6 = Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c6)

    u7 = concatenate([u7, c3])
    c7 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c7)

    u8 = concatenate([u8, c2])
    c8 = Conv2D(
        32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u8)
    c8 = Dropout(0.2)(c8)
    c8 = Conv2D(
        32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c8)

    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(
        16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u9)
    c9 = Dropout(0.2)(c9)
    c9 = Conv2D(
        16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c9)

    decoder_out = Conv2D(1, (1, 1), activation="sigmoid", name="decoder_output")(c9)

    model = Model([input_tensor, input_vector], [decoder_out, classification_output])
    return model
