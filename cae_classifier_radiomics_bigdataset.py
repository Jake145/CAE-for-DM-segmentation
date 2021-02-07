"""docstring"""
import argparse
import concurrent.futures
import logging
import os
from skimage.io import imread
from skimage.transform import resize
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.decomposition import PCA
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from functioncae import cae_cnn_models, caehelper, classes_cae

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

file_handler = logging.FileHandler("BigRadiomicsSegm.log")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


def dice_big(  # pylint: disable=R0913
    list_, k=1, mod=model_rad, shape=(1024, 768, 1), alpha=0.1, lists=dices
):
    """calcola il dice di una singola immagine
    :type list_: lista
    :param list_: lista con il path delle immagini e l'array di feature estratte

    :type k: int
    :param k: valore massimo binarizzato della maschera

    :type mod: keras model
    :param mod: modello di keras precedentemente allenato

    :type shape: array
    :param shape: dimensione di resize dell'immagine

    :type alpha: float
    :param alpha: valore di binarizzazione

    :type lists: list
    :param lists: lista vuota da appendere

    :returns: l'indice di dice
    :rtype: float
    """
    pred = resize(imread(str(list_[0])), shape)
    pred = (
        mod.predict([pred[np.newaxis, ...], list_[1][np.newaxis, ...]])[0].squeeze()
        > alpha
    )
    true = resize(imread(str(list_[2])), shape).squeeze()
    intersection = np.sum(pred[true == k]) * 2.0
    try:
        dice = intersection / (np.sum(pred.squeeze()) + np.sum(true))
    except ZeroDivisionError:
        logger.exception("provato a dividere per zero!")
    logger.info("calcolato correttamente il dice ottenendo %d", dice)
    lists.append(dice)
    return dice


def ypred_creator(  # pylint: disable=R0913
    list_, mod=model_rad, list_app=ypred, shape=(1024, 768, 1)
):
    """calcola le predizioni di classificazione
    :type list_: lista
    :param list_: lista con il path delle immagini e l'array di feature estratte

    :type list_app: list
    :param list_app: lista vuota da appendere

    :type shape: array
    :param shape: dimensione di resize dell'immagine



    :returns: le probabilità di predizione della classificazione
    :rtype: list


    """
    input_ = resize(imread(str(list_[0])), shape)
    output_ = mod.predict([input_[np.newaxis, ...], list_[1][np.newaxis, ...]])[
        1
    ][
        0
    ]
    list_app.append(output_)
    return output_


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Allena la rete con classificazione radiomica e grande dataset. "
    )
    parser.add_argument(
        "-dp",
        "--datapath",
        metavar="",
        help="percorso della cartella dove vi sono le cartelle con le immagini e le maschere",
        default="E:",
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
        "-stp",
        "--steps",
        metavar="",
        type=int,
        help="step per epoca",
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
    args = parser.parse_args()

    DATAPATH = args.datapath
    MAINDIR = "Mass_data_new"
    MASSTRAIN = "Train_data"
    MASKTRAINRES = "resized_masks"
    BENIGN_LABEL = "BENIGN"
    MALIGN_LABEK = "MALIGNANT"

    path_mass_tr = os.path.join(DATAPATH, MAINDIR, MASSTRAIN)
    path_masks_resized = os.path.join(DATAPATH, MAINDIR, MASKTRAINRES)

    images_big_train, masks_big_train, class_big_train = caehelper.read_dataset_big(
        path_mass_tr, path_masks_resized, BENIGN_LABEL, MALIGN_LABEK
    )

    Pandatabigframe = pd.read_csv(args.dataframe)

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

    pca = PCA(n_components=3)
    feature_train_bigg = pca.fit_transform(feature_train_bigg)
    feature_test_bigg = pca.transform(feature_test_bigg)

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
    )

    batch = mass_gen_rad_big[67]

    Validation_data = classes_cae.ValidatorGenerator(
        images_train_rad_big_val,
        masks_train_rad_big_val,
        class_train_rad_big_val,
        feature_train_big_val,
    )

    batch = Validation_data[0]

    model_rad = cae_cnn_models.make_model_rad_big_unet(shape_tensor=(1024, 768, 1))
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
            "classification_output": "categorical_crossentropy",
        },
        metrics={
            "decoder_output": "MAE",
            "classification_output": tf.keras.metrics.AUC(),
        },
    )

    HISTORY_RAD = model_rad.fit(
        mass_gen_rad_big,
        steps_per_epoch=args.steps,
        epochs=args.epocs,
        validation_data=Validation_data,
        callbacks=[model_checkpoint_callback],
    )

    if args.save:
        model_rad.save("model_big_radiomics")

    caehelper.modelviewer(HISTORY_RAD)

    dices = []

    listdicer = []
    for i, _ in enumerate(images_test_rad_big):
        listdicer.append(
            [images_test_rad_big[i], feature_test_bigg[i], masks_test_rad_big[i]]
        )

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.map(dice_big, listdicer)
        print(future)

    dices = np.array(dices)

    meandice = dices.mean()
    logger.info("dice calcolato:%d", meandice)
    ypred = []

    listrocer = []
    for i, _ in enumerate(images_test_rad_big):
        listrocer.append([images_test_rad_big[i], feature_test_bigg[i]])

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.map(ypred_creator, listrocer)
        print(future)

    fpr, tpr, thresholds = roc_curve(
        class_test_rad_big, [item[0] for _, item in enumerate(ypred)], pos_label=0
    )

    auc = auc(fpr, tpr)
    logger.info("auc calcolato:%d", auc)

    caehelper.plot_roc_curve(fpr, tpr, auc)
