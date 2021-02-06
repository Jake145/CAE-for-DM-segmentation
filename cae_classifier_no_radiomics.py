"""docstring"""
import logging
import os

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split

from functioncae import cae_cnn_models, caehelper, classes_cae

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

file_handler = logging.FileHandler("RadiomicsSegm.log")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


DATAPATH = (
    "C:/Users/pensa/Desktop/CAE-for-DM-segmentation/large_sample_Im_segmented_ref"
)

images, masks, labels = caehelper.read_dataset(
    DATAPATH, "pgm", "_2_resized", "_1_resized"
)
images = images / 255
masks = masks / 255


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
# transform
(
    mass_train,
    mass_test,
    mask_train,
    mask_test,
    class_train,
    class_test,
) = train_test_split(images, masks, labels, test_size=0.2, random_state=42)

(
    mass_train_tr,
    mass_train_val,
    mask_train_tr,
    mask_train_val,
    class_train_tr,
    class_train_val,
) = train_test_split(
    mass_train,
    mask_train,
    to_categorical(class_train, 2),
    test_size=0.2,
    random_state=24,
)

mass_gen = classes_cae.MassesSequence(
    mass_train_tr,
    mask_train_tr,
    class_train_tr,
    train_datagen,
)
batch = mass_gen[6]  # define one to get shapes


##
model = cae_cnn_models.make_model_regulizer(
    batch[0][0].shape[1:],
)
model.summary()
MAINPATH = "C:/Users/pensa/Desktop/CAE-for-DM-segmentation/models/"
FILENAME = "base_reg_weights.{epoch:02d}-{val_loss:.2f}.h5"
checkpoint_filepath = os.path.join(MAINPATH, FILENAME)
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor="val_classification_output_auc",
    mode="max",
    save_best_only=True,
)

model.compile(
    optimizer="adam",
    loss={
        "decoder_output": "binary_crossentropy",
        "classification_output": "categorical_crossentropy",
    },
    metrics={"decoder_output": "MAE", "classification_output": tf.keras.metrics.AUC()},
)

EPOCH_NUMBER = 100

HISTORY = model.fit(
    mass_gen,
    steps_per_epoch=len(mass_gen),
    epochs=EPOCH_NUMBER,
    validation_data=(
        mass_train_val,
        [mask_train_val, class_train_val],
    ),
    callbacks=[model_checkpoint_callback],
)

caehelper.modelviewer(HISTORY)


# model = keras.models.load_model(
#    "C:/Users/pensa/Desktop/CAE-for-DM-segmentation/models/rad_Unet_weights.17-0.56.h5"
# )


IDX = 67
xtrain = mass_train[IDX][np.newaxis, ...]
ytrain = mask_train[IDX][np.newaxis, ...]

plt.figure(figsize=(14, 4))
plt.subplot(1, 3, 1)
plt.imshow(xtrain.squeeze())
plt.subplot(1, 3, 2)
plt.imshow(ytrain.squeeze())
plt.subplot(1, 3, 3)
plt.imshow(caehelper.otsu(model.predict(xtrain)[0].squeeze()))


IDX = 16
xtest = mass_test[IDX][np.newaxis, ...]
ytest = mask_test[IDX][np.newaxis, ...]

plt.figure(figsize=(14, 4))
plt.subplot(1, 3, 1)
plt.imshow(xtest.squeeze())
plt.subplot(1, 3, 2)
plt.imshow(ytest.squeeze())
plt.subplot(1, 3, 3)
plt.imshow(caehelper.otsu(model.predict(xtest)[0].squeeze()))


dicetr = caehelper.dice_vectorized(
    mask_train,
    caehelper.otsu(model.predict(mass_train)[0]),
).mean()

docetest = caehelper.dice_vectorized(
    mask_test, caehelper.otsu(model.predict(mass_test)[0])
).mean()

hmap = caehelper.heatmap(mass_test[18], model)

y_pred = model.predict(mass_test)[1]
fpr, tpr, thresholds = roc_curve(
    class_test, [item[0] for _, item in enumerate(y_pred)], pos_label=0
)

auc = auc(fpr, tpr)

caehelper.plot_roc_curve(fpr, tpr, auc)
