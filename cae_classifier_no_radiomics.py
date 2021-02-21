"""docstring"""
import logging
import os
import argparse

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split

from functioncae import cae_cnn_models, caehelper, classes_cae

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Allena la rete base.")
    parser.add_argument(
        "-dp",
        "--datapath",
        metavar="",
        help="percorso della cartella large_sample_Im_segmented_ref",
        default="large_sample_Im_segmented_ref",
    )
    parser.add_argument(
        "-cp",
        "--checkpoint",
        metavar="",
        help="percorso della cartella dove si vuole salvare i checkpoint",
        default="models",
    )
    parser.add_argument(
        "-mod",
        "--model",
        metavar="",
        help="modello da allenare",
        default="regulizer",
        choices=["base", "regulizer", "unet"],
    )
    parser.add_argument(
        "-ep",
        "--epocs",
        metavar="",
        type=int,
        help="epoche dell'allenamento",
        default=100,
    )
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="salva il modello al fine dell'allenamento'",
    )
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

    file_handler = logging.FileHandler("CAESegm.log")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    images, masks, labels = caehelper.read_dataset(
        args.datapath, "pgm", "_2_resized", "_1_resized"
    )
    images = images / 255
    masks = masks / 255

    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
    )

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

    if args.model == "regulizer":

        model = cae_cnn_models.make_model_regulizer(batch[0].shape[1:])
        FILENAME = "reg_weights.{epoch:02d}-{val_loss:.2f}.h5"

    elif args.model == "base":
        model = cae_cnn_models.make_model(batch[0].shape[1:])
        FILENAME = "base_weights.{epoch:02d}-{val_loss:.2f}.h5"

    elif args.model == "unet":
        model = cae_cnn_models.make_model_unet(batch[0].shape[1:])
        FILENAME = "unet_weights.{epoch:02d}-{val_loss:.2f}.h5"

    model.summary()
    MAINPATH = args.checkpoint

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
            "classification_output": "binary_crossentropy",
        },
        metrics={
            "decoder_output": "MAE",
            "classification_output": tf.keras.metrics.AUC(),
        },
    )

    HISTORY = model.fit(
        mass_gen,
        steps_per_epoch=len(mass_gen),
        epochs=args.epocs,
        validation_data=(
            mass_train_val,
            [mask_train_val, class_train_val],
        ),
        callbacks=[model_checkpoint_callback],
    )

    caehelper.modelviewer(HISTORY)

    if args.save:
        model.save("model_no_radiomics")

    IDX = 67
    xtrain = mass_train[IDX][np.newaxis, ...]
    ytrain = mask_train[IDX][np.newaxis, ...]

    plt.figure(figsize=(14, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(xtrain.squeeze())
    plt.subplot(1, 3, 2)
    plt.imshow(ytrain.squeeze())
    plt.subplot(1, 3, 3)
    plt.imshow(model.predict(xtrain)[0].squeeze()>0.1)

    IDX = 16
    xtest = mass_test[IDX][np.newaxis, ...]
    ytest = mask_test[IDX][np.newaxis, ...]

    plt.figure(figsize=(14, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(xtest.squeeze())
    plt.subplot(1, 3, 2)
    plt.imshow(ytest.squeeze())
    plt.subplot(1, 3, 3)
    plt.imshow(model.predict(xtest)[0].squeeze()>0.1)

    dicetr = caehelper.dice_vectorized(
        mask_train,
        model.predict(mass_train)[0]>0.1,
    ).mean()

    docetest = caehelper.dice_vectorized(
        mask_test, model.predict(mass_test)[0]>0.1
    ).mean()

    hmap = caehelper.heatmap(mass_test[18], model)

    y_pred = model.predict(mass_test)[1]
    fpr, tpr, thresholds = roc_curve(
        class_test, [item[0] for _, item in enumerate(y_pred)], pos_label=0
    )

    auc = auc(fpr, tpr)

    caehelper.plot_roc_curve(fpr, tpr, auc)
