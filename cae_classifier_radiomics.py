"""docstring"""
import argparse
import glob
import logging
import os
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from radiomics import featureextractor
from sklearn.decomposition import PCA
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from functioncae import classes_cae, caehelper, cae_cnn_models

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Allena la rete con estrazione delle feature con pyradiomics. "
    )
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
        "-pca",
        "--principalcomponents",
        metavar="",
        type=int,
        help="Numero di componenti principali",
        default=3,
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

    file_handler = logging.FileHandler("RadiomicsSegm.log")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    images_rad, masks_rad, labels_rad = caehelper.read_dataset(
        args.datapath, "pgm", "_2_resized", "_1_resized"
    )
    images_rad = images_rad / 255
    masks_rad = masks_rad / 255

    IMAGES_ID = "_resized"
    MASKS_ID = "_mass_mask"
    EXT = "pgm"
    fnames = glob.glob(os.path.join(args.datapath, f"*{IMAGES_ID}.{EXT}"))
    fnamesmask = glob.glob(os.path.join(args.datapath, f"*{MASKS_ID}.{EXT}"))

    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("gldm")
    extractor.enableFeatureClassByName("glcm")
    extractor.enableFeatureClassByName("shape2D")
    extractor.enableFeatureClassByName("firstorder")
    extractor.enableFeatureClassByName("glrlm")
    extractor.enableFeatureClassByName("glszm")
    extractor.enableFeatureClassByName("ngtdm")

    dataframe = {
        f.replace(args.datapath, ""): extractor.execute(
            caehelper.read_pgm_as_sitk(f),
            caehelper.read_pgm_as_sitk(f.replace(IMAGES_ID, MASKS_ID)),
            label=255,
        )
        for f in fnames
    }

    Pandata = pd.DataFrame(dataframe)

    Pandataframe = Pandata.drop(Pandata.index[0:22]).T

    (
        mass_train_rad,
        mass_test_rad,
        mask_train_rad,
        mask_test_rad,
        class_train_rad,
        class_test_rad,
        feature_train,
        feature_test,
    ) = train_test_split(
        images_rad, masks_rad, labels_rad, Pandataframe, test_size=0.2, random_state=42
    )

    sc = StandardScaler()
    feature_train = sc.fit_transform(feature_train)
    feature_test = sc.transform(feature_test)

    pca = PCA()
    feature_train = pca.fit_transform(feature_train)
    feature_test = pca.transform(feature_test)
    explained_variance = pca.explained_variance_ratio_

    percentage_var_explained = pca.explained_variance_ratio_
    cum_var_explained = np.cumsum(percentage_var_explained)
    # plot spettro della PCA
    plt.figure(figsize=(6, 4))
    plt.clf()
    plt.plot(cum_var_explained, linewidth=2)
    plt.axis("tight")
    plt.grid()
    plt.xlabel("n_components")
    plt.ylabel("Cumulative_Variance_explained")
    plt.show()
    plt.close()

    exp_var_cumsum = pd.Series(
        np.round(pca.explained_variance_ratio_.cumsum(), 4) * 100
    )
    for index, var in enumerate(exp_var_cumsum):
        print("if n_components= %d,   variance=%f" % (index, np.round(var, 3)))

    pca = PCA(n_components=args.principalcomponents)
    feature_train = pca.fit_transform(feature_train)
    feature_test = pca.transform(feature_test)

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

    (
        mass_train_rad_tr,
        mass_train_rad_val,
        mask_train_rad_tr,
        mask_train_rad_val,
        class_train_rad_tr,
        class_train_rad_val,
        feature_train_tr,
        feature_train_val,
    ) = train_test_split(
        mass_train_rad,
        mask_train_rad,
        to_categorical(class_train_rad, 2),
        feature_train,
        test_size=0.2,
        random_state=24,
    )

    mass_gen_rad = classes_cae.MassesSequenceRadiomics(
        mass_train_rad_tr,
        mask_train_rad_tr,
        class_train_rad_tr,
        feature_train_tr,
        train_datagen,
    )
    batch = mass_gen_rad[6]  # define one to get shapes

    if args.model == "regulizer":

        model_rad = cae_cnn_models.make_model_rad_regulizer(
            batch[0][0].shape[1:], batch[0][1].shape[1:]
        )
        FILENAME = "rad_reg_weights.{epoch:02d}-{val_loss:.2f}.h5"

    elif args.model == "base":
        model_rad = cae_cnn_models.make_model_rad(
            batch[0][0].shape[1:], batch[0][1].shape[1:]
        )
        FILENAME = "rad_base_weights.{epoch:02d}-{val_loss:.2f}.h5"

    elif args.model == "unet":
        model_rad = cae_cnn_models.make_model_rad_unet(
            batch[0][0].shape[1:], batch[0][1].shape[1:]
        )
        FILENAME = "rad_unet_weights.{epoch:02d}-{val_loss:.2f}.h5"

    model_rad.summary()
    MAINPATH = args.checkpoint

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
        mass_gen_rad,
        steps_per_epoch=len(mass_gen_rad),
        epochs=args.epocs,
        validation_data=(
            [mass_train_rad_val, feature_train_val],
            [mask_train_rad_val, class_train_rad_val],
        ),
        callbacks=[model_checkpoint_callback],
    )

    caehelper.modelviewer(HISTORY_RAD)
    if args.save:
        model_rad.save("model_radiomics")

    IDX = 67
    xtrain = mass_train_rad[IDX][np.newaxis, ...]
    ytrain = mask_train_rad[IDX][np.newaxis, ...]

    plt.figure(figsize=(14, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(xtrain.squeeze())
    plt.subplot(1, 3, 2)
    plt.imshow(ytrain.squeeze())
    plt.subplot(1, 3, 3)
    plt.imshow(
        caehelper.otsu(
            model_rad.predict([xtrain, feature_train[IDX][np.newaxis, ...]])[
                0
            ].squeeze()
        )
    )

    IDX = 16
    xtest = mass_test_rad[IDX][np.newaxis, ...]
    ytest = mask_test_rad[IDX][np.newaxis, ...]

    plt.figure(figsize=(14, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(xtest.squeeze())
    plt.subplot(1, 3, 2)
    plt.imshow(ytest.squeeze())
    plt.subplot(1, 3, 3)
    plt.imshow(
        caehelper.otsu(
            model_rad.predict([xtest, feature_test[IDX][np.newaxis, ...]])[0].squeeze()
        )
    )

    dicetr = caehelper.dice_vectorized(
        mask_train_rad,
        caehelper.otsu(model_rad.predict([mass_train_rad, feature_train])[0]),
    ).mean()

    docetest = caehelper.dice_vectorized(
        mask_test_rad,
        caehelper.otsu(model_rad.predict([mass_test_rad, feature_test])[0]),
    ).mean()

    hmap = caehelper.heatmap_rad(mass_test_rad[16], feature_test[16], model_rad)

    y_pred = model_rad.predict([mass_test_rad, feature_test])[1]
    fpr, tpr, thresholds = roc_curve(
        class_test_rad, [item[0] for _, item in enumerate(y_pred)], pos_label=0
    )

    auc = auc(fpr, tpr)

    caehelper.plot_roc_curve(fpr, tpr, auc)
