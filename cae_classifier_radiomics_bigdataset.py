"""Implementazione di CAE con feature radiomiche per il dataset TCIA"""
import argparse
import os

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from functioncae import cae_cnn_models, caehelper, classes_cae

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Allena la rete con classificazione radiomica e grande dataset. "
    )
    parser.add_argument(
        "-dp",
        "--datapath",
        metavar="",
        help="percorso della cartella dove vi sono le cartelle con le immagini e le maschere",
        default="E:/Test2",
    )
    parser.add_argument(
        "-cp",
        "--checkpoint",
        metavar="",
        help="percorso della cartella dove si vuole salvare i checkpoint",
        default="models",
    )
    parser.add_argument(
        "-ep",
        "--epocs",
        metavar="",
        type=int,
        help="epoche dell'allenamento",
        default=10,
    )
    parser.add_argument(
        "-df",
        "--dataframe",
        metavar="",
        help="percorso dove si trova il .csv con le feature",
        default="Pandatabigframe.csv",
    )
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="salva il modello al fine dell'allenamento'",
    )
    parser.add_argument(
        "-t",
        "--tensor",
        help="Tensore di Reshape, scrivi tre interi separati da spazio",
        metavar="",
        nargs="+",
        type=int,
        default=[
            1024,
            768,
        ],
    )
    parser.add_argument(
        "-b",
        "--batch",
        metavar="",
        type=int,
        help="dimensione della batch",
        default=10,
    )
    parser.add_argument(
        "-pca",
        "--principalcomponents",
        metavar="",
        type=int,
        help="Numero di componenti principali",
        default=3,
    )

    args = parser.parse_args()

    DATAPATH = args.datapath
    MASSTRAIN = "Train_data"
    MASKTRAINRES = "resized_masks"
    BENIGN_LABEL = "BENIGN"
    MALIGN_LABEK = "MALIGNANT"

    path_mass_tr = os.path.join(DATAPATH, MASSTRAIN)
    path_masks_resized = os.path.join(DATAPATH, MASKTRAINRES)

    images_big_train, masks_big_train, class_big_train = caehelper.read_dataset_big(
        path_mass_tr, path_masks_resized, BENIGN_LABEL, MALIGN_LABEK
    )

    Pandatabigframe = pd.read_csv(args.dataframe)

    Pandatabigframe.sort_values(by="Unnamed: 0")

    Pandatabigframe = Pandatabigframe.iloc[:, 1:]

    (
        images_train_rad_big,
        images_test_rad_big,
        masks_train_rad_big,
        masks_test_rad_big,
        class_train_rad_big,
        class_test_rad_big,
        feature_train_big,
        feature_test_big,
    ) = train_test_split(
        images_big_train,
        masks_big_train,
        class_big_train,
        Pandatabigframe,
        test_size=0.2,
        random_state=42,
    )

    sc = StandardScaler()
    feature_train_bigg = sc.fit_transform(feature_train_big)
    feature_test_bigg = sc.transform(feature_test_big)

    pca = PCA()
    feature_train_bigg = pca.fit_transform(feature_train_bigg)
    feature_test_bigg = pca.transform(feature_test_bigg)
    explained_variance_big = pca.explained_variance_ratio_

    percentage_var_explained = pca.explained_variance_ratio_
    cum_var_explained = np.cumsum(percentage_var_explained)
    # plot spettro dell pca
    plt.figure(1, figsize=(6, 4))
    plt.clf()
    plt.plot(cum_var_explained, linewidth=2)
    plt.axis("tight")
    plt.grid()
    plt.xlabel("n_components")
    plt.ylabel("Cumulative_Variance_explained")
    plt.show()

    exp_var_cumsum = pd.Series(
        np.round(pca.explained_variance_ratio_.cumsum(), 4) * 100
    )
    for index, var in enumerate(exp_var_cumsum):
        print("if n_components= %d,   variance=%f" % (index, np.round(var, 3)))

    pca = PCA(n_components=args.principalcomponents)
    feature_train_bigg = pca.fit_transform(feature_train_bigg)
    feature_test_bigg = pca.transform(feature_test_bigg)

    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
    )

    (
        images_train_rad_big_tr,
        images_train_rad_big_val,
        masks_train_rad_big_tr,
        masks_train_rad_big_val,
        class_train_rad_big_tr,
        class_train_rad_big_val,
        feature_train_big_tr,
        feature_train_big_val,
    ) = train_test_split(
        images_train_rad_big,
        masks_train_rad_big,
        to_categorical(class_train_rad_big, 2),
        feature_train_bigg,
        test_size=0.2,
        random_state=42,
    )

    mass_gen_rad_big = classes_cae.MassesSequenceRadiomicsBig(
        images_train_rad_big_tr,
        masks_train_rad_big_tr,
        class_train_rad_big_tr,
        feature_train_big_tr,
        train_datagen,
        batch_size=args.batch,
        shape_tensor=tuple(args.tensor),
    )

    # batch = mass_gen_rad_big[67]

    Validation_data = classes_cae.ValidatorGenerator(
        images_train_rad_big_val,
        masks_train_rad_big_val,
        class_train_rad_big_val,
        feature_train_big_val,
        batch_size=args.batch,
        shape_tensor=tuple(args.tensor),
    )

    model_rad = cae_cnn_models.make_model_rad_big_unet(
        shape_tensor=tuple(args.tensor), feature_dim=(args.principalcomponents,)
    )
    model_rad.summary()
    MAINPATH = args.checkpoint
    FILENAME = "big_weights.{epoch:02d}-{val_loss:.2f}.h5"
    checkpoint_filepath = os.path.join(MAINPATH, FILENAME)
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor="val_classification_output_auc",
        mode="max",
        save_best_only=True,
    )

    model_rad.compile(
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

    HISTORY_RAD = model_rad.fit(
        mass_gen_rad_big,
        steps_per_epoch=len(mass_gen_rad_big),
        epochs=args.epocs,
        validation_data=Validation_data,
        callbacks=[model_checkpoint_callback],
    )

    if args.save:
        model_rad.save("model_big_radiomics")

    caehelper.modelviewer(HISTORY_RAD)
