import concurrent.futures
import glob
import logging
import os
import pickle
import re
import sys
import time
import warnings
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import radiomics
import SimpleITK as sitk
from PIL import Image
from radiomics import featureextractor


from functioncae import ClassesCAE, caehelper

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

file_handler = logging.FileHandler("Feature_extraction.log")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


def resizer(list, endpath, pattern):
    """Funzione per fare il reshape delle maschere in maniera che combacino con le immagini a cui sono riferite

    type list: lista
    param list: lista che contiene il path della immagine e il path della maschera

    type endpath: stringa
    param endpath: path di arrivo per le nuove maschere

    type pattern: stringa
    param pattern: pattern da trovare per dare il nome corretto alla nuova maschera. Usa regex
    """
    logger.info(
        f"sto cercando di leggere come immagine {list[0]} e maschera {list[1]} da salvare in {endpath} con pattern da trovare {pattern}"
    )
    a = time.perf_counter()
    try:
        image = Image.open(list[0])
    except:
        warnings.warn(f"Immagine {list[0]} mancante o corrotta")
        logger.exception(
            f"Immagine {list[0]} mancante o corrotta, non riesco a leggerla"
        )

    try:
        mask = Image.open(list[1])
    except:
        warnings.warn(f"Immagine {list[1]} mancante o corrotta")
        logger.exception(
            f"maschera {list[1]} mancante o corrotta, non riesco a leggerla"
        )

    try:

        mask = mask.resize(image.size)
        logger.debug(
            f"ho fatto il resize di {list[1]} usando come dimensione {image.shape} di {list[0]}"
        )
    except:
        warnings.warn(
            "Non riesco a fare il resize. Sto salvando l'immagine senza fare resize!!!"
        )

        logger.critical(
            "Non riesco a fare il resize. Sto salvando l'immagine senza fare resize!!!"
        )
    try:
        match = re.findall(pattern, list[0])[0]
        logger.debug(f"il match del pattern è {match}")
        filename = os.path.join(endpath, match + ".png")
        mask.save(filename)
        logger.info("salvata la nuova maschera in {filename}")

    except:
        warnings.warn("Non possibile andare avanti")
        logger.warning(
            "Non possibile andare avanti, non trovo il pattern o non riesco a salvare il file"
        )
    b = time.perf_counter()

    logger.info(f"time elapsed: {b-a}")
    return filename


##
if __name__ == "__main__":

    datapath = "E:"
    maindir = "Mass_data_new"
    mass_train = "Train_data"
    mass_test = "Test_data"
    mask_train = "Train_data_masks"
    mask_test = "Test_data_masks"
    mask_train_res = "resized_masks"
    benign_label = "BENIGN"
    malign_label = "MALIGNANT"
    features_path = "feats"

    path_mass_tr = os.path.join(datapath, maindir, mass_train)
    path_masks_tr = os.path.join(datapath, maindir, mask_train)

    path_masks_resized = os.path.join(datapath, maindir, mask_train_res)

    X_big_train, Y_big_train, Class_big_train = read_dataset_big(
        path_mass_tr, path_masks_tr, benign_label, malign_label
    )

    if not os.path.exists(path_masks_resized):
        os.makedirs(path_masks_resized)
        logger.info(f"creato il path {path_masks_resized}")

    ##
    """#Pyradiomics on big dataset"""

    endpath_tr = os.path.join(datapath, maindir, features_path)
    if not os.path.exists(endpath_tr):
        os.makedirs(endpath_tr)
        logger.info(f"creato il path {endpath_tr}")

    ##

    """Questa funzione serve per estrarre le feature in multiprocessing e aggiungerle a un dizionario"""

    extractor = featureextractor.RadiomicsFeatureExtractor()
    logger.info(f"inizializzato estrattore di pyradiomics")

    ##
    biggy = [[X_big_train[i], Y_big_train[i]] for i in range(len(X_big_train))]
    # biggy_test=[[X_big_test[i],Y_big_test[i]] for i in range(len(X_big_test))]

    #
    ##
    rez = partial(
        resizer,
        endpath=path_masks_resized,
        pattern=re.compile(r"[M][\w-]*[0-9]*[\w]{13}"),
    )
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor() as executor:

        results = executor.map(rez, biggy)
        logger.debug(f"{results}")
        print(results)
    end = time.perf_counter()

    logger.info(f"Elapsed time for MT:{end-start}")

    ##

    X_big_train_n, Y_big_train_n, Class_big_train_n = read_dataset_big(
        path_mass_tr, path_masks_resized, benign_label, malign_label
    )

    biggy = [[X_big_train_n[i], Y_big_train_n[i]] for i in range(len(X_big_train_n))]

    ##

    # nam=radiomic_dooer([['E:Mass_data_new\\Train_data\\Mass-Training_P_01152_RIGHT_MLO_BENIGN.png', 'E:Mass_data_new\\resized_masks\\Mass-Training_P_01152_RIGHT_MLO_BENIGN.png']],path_mass_tr,endpath_tr,255,extractor)
    # os.remove('E:Mass_data_new\\Train_data\\Mass-Training_P_01152_RIGHT_MLO_BENIGN.png')
    # os.remove('E:Mass_data_new\\resized_masks\\Mass-Training_P_01152_RIGHT_MLO_BENIGN.png')

    ##

    ##
    # this is the filename list for the multiprocessing
    errors = []
    for item in biggy:
        try:
            name = radiomic_dooer(item, path_mass_tr, endpath_tr, 255, extractor)
        except:
            logger.debug(f"un file errato: {path_mass_tr}")
            errors.append(item)
    ##
    diz = {}
    diz_up = partial(dict_update_radiomics, dictionary=diz)
    liz = []
    list_items = next(os.walk(endpath_tr))[2]
    for items in list_items:
        liz.append(os.path.join(endpath_tr, items))
    ##
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor() as executor:

        results = executor.map(diz_up, liz)
        logger.debug(f"{results}")
        print(results)
    end = time.perf_counter()

    logger.info(f"Elapsed time for MT:{end-start}")

    import pandas as pd

    Pandata_big = pd.DataFrame(diz)

    for i, name in enumerate(Pandata_big.index):
        if "diagnostics" in Pandata_big.index[i]:
            print(i)
        else:
            pass
    ##

    # Pandatabigframe=Pandata_big.drop(Pandata_big.index[0:22]).T

    gfg_csv_data = Pandatabigframe.to_csv(
        "C:/Users/pensa/Desktop/CAE-for-DM-segmentation/Pandatabigframe.csv", index=True
    )

    from sklearn.model_selection import train_test_split

    (
        X_train_rad_big,
        X_test_rad_big,
        Y_train_rad_big,
        Y_test_rad_big,
        class_train_rad_big,
        class_test_rad_big,
        feature_train_big,
        feature_test_big,
    ) = train_test_split(
        X_big_train_n,
        Y_big_train_n,
        Class_big_train_n,
        Pandatabigframe,
        test_size=0.2,
        random_state=42,
    )

    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    feature_train_bigg = sc.fit_transform(feature_train_big)
    feature_test_bigg = sc.transform(feature_test_big)

    from sklearn.decomposition import PCA

    pca = PCA()
    feature_train_bigg = pca.fit_transform(feature_train_bigg)
    feature_test_bigg = pca.transform(feature_test_bigg)
    explained_variance_big = pca.explained_variance_ratio_

    explained_variance_big

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

    from sklearn.decomposition import PCA

    pca = PCA(n_components=3)
    feature_train_bigg = pca.fit_transform(feature_train_bigg)
    feature_test_bigg = pca.transform(feature_test_bigg)

    import keras
    from keras.preprocessing.image import ImageDataGenerator
    from skimage.transform import resize
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

    class MassesSequenceRadiomicsBig(keras.utils.Sequence):
        """ Classe per data augmentation per CAE con grandi dati """

        def __init__(
            self,
            x,
            y,
            label_array,
            features,
            img_gen,
            batch_size=5,
            shape=(2048, 1536),
            shape_tensor=(2048 / 2, 1536 / 2, 1),
        ):
            """Inizializza la sequenza

            Parametri:

            x (np.array): path delle immagini
            y (np.array): path delle maschere
            label_array (np.array): label di classificazione
            features (np.array): array di feature dopo la pca
            batch_size (int): dimensione della batch
            img_gen (ImageDatagenerator): Una istanza della classe ImageDatagenerator
            shape (tuple): shape dell'immagine. Di Default è (2048, 1536) per il limite di colab, per la Unet invece è metà di queste.

            """
            self.x, self.y, self.label_array, self.features = (
                x,
                y,
                label_array,
                features,
            )
            self.shape = shape
            self.shape_tensor = shape_tensor
            self.img_gen = img_gen
            self.batch_size = batch_size

        def __len__(self):
            return len(self.x) // self.batch_size

        def on_epoch_end(self):
            """Shuffle the dataset at the end of each epoch."""
            self.x, self.y, self.label_array, self.features = shuffle(
                self.x, self.y, self.label_array, self.features
            )

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
            batch_features = self.features[
                idx * self.batch_size : (idx + 1) * self.batch_size
            ]

            X = []
            Y = []
            Classes = []
            Features = []

            for image, mask, label, feature in zip(
                batch_x, batch_y, batch_label_array, batch_features
            ):
                transform = self.img_gen.get_random_transform(self.shape)
                x_el = resize(imread(str(image)), self.shape_tensor)
                Y_el = resize(imread(str(mask)), self.shape_tensor)
                X.append(self.process(x_el, transform))
                del x_el
                Y.append(self.process(Y_el, transform))
                del Y_el
                Classes.append(label)
                Features.append(feature)

            return [np.array(X) / 255, np.asarray(Features, np.float64)], [
                np.array(Y) / 255,
                np.asarray(Classes, np.float),
            ]

    ##

    from keras.utils import to_categorical

    (
        X_train_rad_big_tr,
        X_train_rad_big_val,
        Y_train_rad_big_tr,
        Y_train_rad_big_val,
        class_train_rad_big_tr,
        class_train_rad_big_val,
        feature_train_big_tr,
        feature_train_big_val,
    ) = train_test_split(
        X_train_rad_big,
        Y_train_rad_big,
        to_categorical(class_train_rad_big, 2),
        feature_train_bigg,
        test_size=0.2,
        random_state=24,
    )

    mass_gen_rad_big = MassesSequenceRadiomicsBig(
        X_train_rad_big_tr,
        Y_train_rad_big_tr,
        class_train_rad_big_tr,
        feature_train_big_tr,
        train_datagen,
    )

    batch = mass_gen_rad_big[67]

    # del(batch)

    batch[0][0].shape
    ##
    class Validator_Generator(keras.utils.Sequence):
        def __init__(
            self,
            x,
            y,
            label_array,
            features,
            batch_size=5,
            shape=(2048, 1536),
            shape_tensor=(2048 / 2, 1536 / 2, 1),
        ):
            """Inizializza la sequenza

            Parametri:

            x (np.array): path delle immagini
            y (np.array): path delle maschere
            label_array (np.array): label di classificazione
            features (np.array): array di feature dopo la pca
            batch_size (int): dimensione della batch
            img_gen (ImageDatagenerator): Una istanza della classe ImageDatagenerator
            shape (tuple): shape dell'immagine. Di Default è (2048, 1536) per il limite di colab, per la Unet invece è metà di queste.

            """
            self.x, self.y, self.label_array, self.features = (
                x,
                y,
                label_array,
                features,
            )
            self.shape = shape
            self.batch_size = batch_size
            self.shape_tensor = shape_tensor

        def __len__(self):
            return len(self.x) // self.batch_size

        def __getitem__(self, idx):
            batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]
            batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]
            batch_label_array = self.label_array[
                idx * self.batch_size : (idx + 1) * self.batch_size
            ]
            batch_features = self.features[
                idx * self.batch_size : (idx + 1) * self.batch_size
            ]

            X = []
            Y = []
            Classes = []
            Features = []

            for image, mask, label, feature in zip(
                batch_x, batch_y, batch_label_array, batch_features
            ):

                x_el = resize(imread(str(image)), self.shape_tensor)
                Y_el = resize(imread(str(mask)), self.shape_tensor)
                X.append(x_el)
                del x_el
                Y.append(Y_el)
                del Y_el
                Classes.append(label)
                Features.append(feature)

            return [np.array(X) / 255, np.asarray(Features, np.float64)], [
                np.array(Y),
                np.asarray(Classes, np.float),
            ]

    Validation_data = Validator_Generator(
        X_train_rad_big_val,
        Y_train_rad_big_val,
        class_train_rad_big_val,
        feature_train_big_val,
    )

    ##

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

    def make_model_rad_BIG_REGULIZER(
        shape_tensor=(4096, 3072, 1), feature_dim=feature_train_big_tr.shape[1:]
    ):
        input_tensor = Input(shape=shape_tensor, name="tensor_input")
        input_vector = Input(shape=feature_dim)

        x = Conv2D(32, (5, 5), strides=2, padding="same", activation="relu")(
            input_tensor
        )
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
        model = Model(
            [input_tensor, input_vector], [decoder_out, classification_output]
        )

        return model

    def make_model_rad_BIG(
        shape_tensor=(2048, 1536, 1), feature_dim=feature_train_big_tr.shape[1:]
    ):
        input_tensor = Input(shape=shape_tensor)
        input_vector = Input(shape=feature_dim)

        x = Conv2D(32, (5, 5), strides=2, padding="same", activation="relu")(
            input_tensor
        )
        # x = Dropout(.2)(x)
        x = Conv2D(64, (3, 3), strides=2, padding="same", activation="relu")(x)
        # x = Dropout(.2)(x)
        x = Conv2D(
            128, (3, 3), strides=2, padding="same", activation="relu", name="last_conv"
        )(x)

        flat = Flatten()(x)
        flat = concatenate([flat, input_vector])
        den = Dense(16, activation="relu")(flat)
        # den = Dropout(.2)(den)

        classification_output = Dense(
            2, activation="sigmoid", name="classification_output"
        )(flat)

        x = Conv2DTranspose(64, (3, 3), strides=2, padding="same", activation="relu")(x)
        x = Conv2DTranspose(32, (3, 3), strides=2, padding="same", activation="relu")(x)
        x = Conv2DTranspose(32, (3, 3), strides=2, padding="same", activation="relu")(x)
        decoder_out = Conv2D(
            1, (1, 1), padding="valid", activation="sigmoid", name="decoder_output"
        )(x)
        model = Model(
            [input_tensor, input_vector], [decoder_out, classification_output]
        )

        return model

    def make_model_rad_BIG_UNET(
        shape_tensor=(1024, 768, 1), feature_dim=feature_train_big_tr.shape[1:]
    ):
        input_tensor = Input(shape=shape_tensor)
        input_vector = Input(shape=feature_dim)

        c1 = Conv2D(
            16,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
        )(input_tensor)
        c1 = Dropout(0.2)(c1)
        c1 = Conv2D(
            16,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
        )(c1)
        p1 = MaxPooling2D((2, 2))(c1)
        c2 = Conv2D(
            32,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
        )(p1)
        c2 = Dropout(0.1)(c2)
        c2 = Conv2D(
            32,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
        )(c2)
        p2 = MaxPooling2D((2, 2))(c2)
        # p2 = Resizing(32,32,interpolation='nearest')(p2)
        c3 = Conv2D(
            64,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
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
        # p3 = Resizing(16,16,interpolation='nearest')(p3)
        c4 = Conv2D(
            128,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
        )(p3)
        c4 = Dropout(0.2)(c4)
        c4 = Conv2D(
            128,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
        )(c4)

        p4 = MaxPooling2D((2, 2))(c4)

        c5 = Conv2D(
            256,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
        )(p4)

        c5 = Dropout(0.2)(c5)
        c5 = Conv2D(
            256,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
        )(c5)
        # fc layers

        flat = Flatten()(c3)
        flat = concatenate([flat, input_vector])
        den = Dense(16, activation="relu")(flat)

        classification_output = Dense(
            2, activation="softmax", name="classification_output"
        )(flat)

        u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c5)

        # c4 = Resizing(14,14,interpolation='nearest')(c4)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(
            128,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
        )(u6)
        c6 = Dropout(0.2)(c6)
        c6 = Conv2D(
            128,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
        )(c6)

        u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c6)
        # c3= Resizing(28,28,interpolation='nearest')(c3)

        u7 = concatenate([u7, c3])
        c7 = Conv2D(
            64,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
        )(u7)
        c7 = Dropout(0.2)(c7)
        c7 = Conv2D(
            64,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
        )(c7)

        u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c7)
        # u8 = Resizing(62,62,interpolation='nearest')(c2)

        u8 = concatenate([u8, c2])
        c8 = Conv2D(
            32,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
        )(u8)
        c8 = Dropout(0.2)(c8)
        c8 = Conv2D(
            32,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
        )(c8)

        u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c8)
        # c1= Resizing(112,112,interpolation='nearest')(c1)

        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(
            16,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
        )(u9)
        c9 = Dropout(0.2)(c9)
        c9 = Conv2D(
            16,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
        )(c9)

        decoder_out = Conv2D(1, (1, 1), activation="sigmoid", name="decoder_output")(c9)

        model = Model(
            [input_tensor, input_vector], [decoder_out, classification_output]
        )
        return model

    ##

    model_rad = make_model_rad_BIG_UNET()
    model_rad.summary()

    checkpoint_filepath = "big_weights.{epoch:02d}-{val_loss:.2f}.h5"
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

    epoch_number = 10

    history_rad = model_rad.fit(
        mass_gen_rad_big,
        steps_per_epoch=10,
        epochs=epoch_number,
        validation_data=Validation_data,
        callbacks=[model_checkpoint_callback],
    )

    modelviewer(history_rad)
