"""#Helper Functions

"""


import glob
import importlib
import os
import pickle
import shutil
import time

import cv2
import matplotlib.pyplot as plt

# In questo file sono riportate tutte le funzioni helper per il software. Farò in un altro i codici per i modelli
##
import numpy as np
import PIL
from keras.utils import to_categorical
from PIL import Image
from skimage import io, transform
from skimage.io import imread
from sklearn.model_selection import train_test_split

# import sys
# sys.path.append('/CAE-for-DM-segmentation/package/')
importlib.import_module("package")
##
imagepath = (
    "/CAE-for-DM-segmentation/large_sample_Im_segmented_ref/0003s1_1_1_1_resized.pgm"
)
masspath = (
    "/CAE-for-DM-segmentation/large_sample_Im_segmented_ref/0003s1_1_1_1_mass_mask.pgm"
)

"""# Possiamo vedere un esempio dei dati a disposizione"""

img = io.imread(masspath)
img2 = io.imread(imagepath)
plt.figure(figsize=(14, 4))
plt.subplot(1, 2, 2)
plt.title("image")
plt.imshow(img2)
plt.subplot(1, 2, 1)
plt.title("real mass")
plt.imshow(img)
plt.show()

"""Importiamo la funzione read_dataset e normalizziamo le immagini"""

datapath = "/content/drive/MyDrive/large_sample_Im_segmented_ref"

X, Y, LABELS = read_dataset(datapath, "pgm", "_2_resized", "_1_resized")
X = X / 255
Y = Y / 255
Y.min(), Y.max()

LABELS

"""Splittiamo il dataset in allenamento e testing"""

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test, class_train, class_test = train_test_split(
    X, Y, LABELS, test_size=0.2, random_state=42
)

"""# Data augmentation: Utilizzando una classe che fa da generatore custom per l'allenamento che eredita da keras.utilis.sequence possiamo via via caricare le batch di immagini alla rete

Fondamentale è la classe ImageDataGenerator , che permette di definire le trasformazioni per fare data augmentation
"""

from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle

train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="reflect",
)

transform = train_datagen.get_random_transform((124, 124))

import keras


class MassesSequence(keras.utils.Sequence):
    """ Classe per fare data augmentation per CAE """

    def __init__(self, x, y, label_array, img_gen, batch_size=10, shape=(124, 124)):
        """

        Parametri:

        x (np.array): immagini
        y (np.array): maschere
        label_array (np.array): label di classificazione (benigno o maligno)
        batch_size (int): dimensione della batch
        img_gen (ImageDatagenerator): istanza della classe ImageDatagenerator
        shape (tuple): dimensione delle immagini. Di default (124, 124)

        """
        self.x, self.y, self.label_array = x, y, label_array
        self.shape = shape
        self.img_gen = img_gen
        self.batch_size = batch_size

    def __len__(self):
        return len(self.x) // self.batch_size

    def on_epoch_end(self):
        """Shuffle the dataset at the end of each epoch."""
        self.x, self.y, self.label_array = shuffle(self.x, self.y, self.label_array)

    def process(self, img, transform):
        """ Apply a transformation to an image """
        img = self.img_gen.apply_transform(img, transform)
        return img

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_label_array = self.label_array[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]

        X = []
        Y = []
        Classes = []

        for image, mask, label in zip(self.x, self.y, self.label_array):
            transform = self.img_gen.get_random_transform(self.shape)
            X.append(self.process(image, transform))
            Y.append(self.process(mask, transform) > 0.2)
            Classes.append(label)

        return np.asarray(X, np.float64), [
            np.asarray(Y, np.float64),
            np.asarray(Classes, np.float),
        ]


"""Avendo definito la classe che restituisce la batch di immagini e la lista di label (maschera e classe) ci rimane da dividere il vettore di training definito prima per ottenere lo split training-validazione. E' in questo momento che si trasformano le classi 0 e 1 in valori di probabilità con to_categorical"""

(
    X_train_tr,
    X_train_val,
    Y_train_tr,
    Y_train_val,
    class_train_tr,
    class_train_val,
) = train_test_split(
    X_train, Y_train, to_categorical(class_train, 2), test_size=0.2, random_state=24
)

mass_gen = MassesSequence(X_train_tr, Y_train_tr, class_train_tr, train_datagen)

batch_example = mass_gen[6]

batch_example[0].shape[1:]

"""#Definiamo i modelli"""

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard

import datetime
import os

import tensorflow as tf
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
from keras.models import Model, load_model

"""Model by Liu et al, Deep Convolutional Auto-Encoder and 3D Deformable Approach for Tissue Segmentation in Magnetic Resonance Imaging, Proc. Intl. Soc. Mag. Reson. Med. 25, 2017"""


def make_model(shape=batch_example[0].shape[1:]):
    input_tensor = Input(shape=shape)

    x = Conv2D(32, (5, 5), strides=2, padding="same", activation="relu")(input_tensor)
    x = Conv2D(64, (3, 3), strides=2, padding="same", activation="relu")(x)
    x = Conv2D(
        128, (3, 3), strides=2, padding="same", activation="relu", name="last_conv"
    )(x)

    flat = Flatten()(x)
    den = Dense(16, activation="relu")(flat)
    classification_output = Dense(
        2, activation="sigmoid", name="classification_output"
    )(flat)

    x = Conv2DTranspose(64, (3, 3), strides=2, padding="same", activation="relu")(x)
    x = Conv2DTranspose(32, (3, 3), strides=2, padding="same", activation="relu")(x)
    x = Conv2DTranspose(32, (3, 3), strides=2, padding="same", activation="relu")(x)
    decoder_out = Conv2D(
        1, (5, 5), padding="valid", activation="sigmoid", name="decoder_output"
    )(x)
    model = Model(input_tensor, [decoder_out, classification_output])

    return model


"""Model 2 is the same but with added regularization (dropout layers) and maxpooling


"""


def make_modelREGULIZER(shape=batch_example[0].shape[1:]):
    input_tensor = Input(shape=shape, name="model_input")

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
    )(flat)

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


"""This model is the Unet from Ronneberger e al, U-Net: Convolutional Networks for Biomedical
Image Segmentation. I added a resizing layer to adapt it for our image size

"""

from keras.constraints import max_norm, min_max_norm, unit_norm
from tensorflow.keras import regularizers


def make_modelUNET(shape=batch_example[0].shape[1:]):
    input_tensor = Input(shape=shape)

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
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="last_conv",
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
        256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c5)
    # fc layers

    flat = Flatten()(c3)
    den = Dense(16, activation="relu")(flat)
    classification_output = Dense(
        2, activation="sigmoid", name="classification_output"
    )(flat)

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


"""#Da qui si importano i modelli e si fa il training di una rete. Per il momento si allena il modello REGULIZER"""

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard

import datetime

import tensorflow as tf

model_1 = make_model()
model_1.summary()

model_REG = make_modelREGULIZER()
model_REG.summary()

model_UNET = make_modelUNET()
model_UNET.summary()

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

"""Definiamo dei checkpoint per avere il modello con la migliore predittività in quanto l'autoencoder riesce a raggiungere prestazioni migliori più facilmente del classificatore."""

checkpoint_filepath = "w_weights.{epoch:02d}-{val_loss:.2f}.h5"
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor="val_classification_output_auc",
    mode="max",
    save_best_only=True,
)

"""Come metrica per la classificazione utilizziamo direttamente l'AUC di Keras"""

model_REG.compile(
    optimizer="adam",
    loss={
        "decoder_output": "binary_crossentropy",
        "classification_output": "categorical_crossentropy",
    },
    metrics={"decoder_output": "MAE", "classification_output": tf.keras.metrics.AUC()},
)

epoch_number = 50

history = model_REG.fit(
    mass_gen,
    steps_per_epoch=len(mass_gen),
    epochs=epoch_number,
    validation_data=(X_train_val, [Y_train_val, class_train_val]),
    callbacks=[tensorboard_callback, model_checkpoint_callback],
)

"""Visualizzazione su tensorboard"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir logs

"""Per avere una visualizzazione rapida delle loss, chiamiamo la funzione modelviewer"""

modelviewer(history)

"""Carichiamo il modello corrispondente all'epoca migliore"""

model_datapath = "/content/w_weights.01-1.29.h5"
model_rad = keras.models.load_model(model_datapath)

"""Guardiamo un esempio di output del CAE"""

idx = 67
xtrain = X_train[idx][np.newaxis, ...]
ytrain = Y_train[idx][np.newaxis, ...]

plt.figure(figsize=(14, 4))
plt.subplot(1, 3, 1)
plt.imshow(xtrain.squeeze())
plt.subplot(1, 3, 2)
plt.imshow(ytrain.squeeze())
plt.subplot(1, 3, 3)
plt.imshow(otsu(model_REG.predict(xtrain)[0].squeeze()))

"""Guardiamo un esempio su immagini di test"""

idx = 16
xtest = X_test[idx][np.newaxis, ...]
ytest = Y_test[idx][np.newaxis, ...]
xtest.shape

plt.figure(figsize=(14, 4))
plt.subplot(1, 3, 1)
plt.imshow(xtest.squeeze())
plt.subplot(1, 3, 2)
plt.imshow(ytest.squeeze())
plt.subplot(1, 3, 3)
plt.imshow(otsu(model_rad.predict(xtest)[0].squeeze()))

"""Per valutare le prestazioni del CAE, calcoliamo l'indice di dice medio sul set di training e sul set di test. Per binarizzare l'output si utilizza l'algoritmo di Otsu."""

dice_vectorized(Y_train, otsu(model_REG.predict(X_train)[0])).mean()

dice_vectorized(Y_test, otsu(model_REG.predict(X_test)[0])).mean()

"""Con la seguente funzione visualizziamo la heatmap dell'ultimo layer convoluzionale prima del layer fully connected del classificatore"""

heat = heatmap(X_test[18], model_REG)

"""Guardiamo ora la curva roc e la AUC sul set di test"""

from sklearn.metrics import roc_curve

y_pred = model_REG.predict(X_test)[1]
fpr, tpr, thresholds = roc_curve(
    class_test, [item[0] for _, item in enumerate(y_pred)], pos_label=0
)

from sklearn.metrics import auc

auc = auc(fpr, tpr)

plot_roc_curve(fpr, tpr, auc)
