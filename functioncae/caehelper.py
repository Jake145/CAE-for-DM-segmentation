"""docstring"""
import glob
import logging
import os
import re
import time
import warnings
import pickle
import SimpleITK as sitk
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.python.keras import backend as K  # pylint: disable=E0611
from skimage.filters import threshold_multiotsu
import matplotlib.pyplot as plt
from PIL import Image
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

file_handler = logging.FileHandler("CAE_functions.log")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


def save_newext(file_name, data_path, ext1, ext2, endpath):
    """Riscrive le immagini in formato leggibile per pyradiomics
    :type file_name: str
    :param file_path: nome del file della immagine

    :type data_path: str
    :param data_path: percorso della cartella dove si trova la immagine

    :type ext1: str
    :param ext1: stringa identificativa dell'estenzione di partenza della immagine

    :type ext2: str
    :param ext2: stringa identificativa dell'estenzione finale della immagine

    :type endpath: str
    :param endpath: percorso della cartella di arrivo
    :returns: dopo aver salvato il nuovo file, restituisce l'esito
    :rtype: bool


    """

    if ext1 == ext2:
        logger.debug(
            "il file %s in %s ha già la estenzione %s", file_name, data_path, ext2
        )
    try:
        image = plt.imread(os.path.join(data_path, file_name))
        file_name = file_name.replace(
            f".{ext1}", f".{ext2}"
        )
        logger.info(
            "read %s and changed extension from %s to%s",
            os.path.join(data_path, file_name),
            ext1,
            ext2,
        )
    except Exception as e_error:
        raise Exception("immagine o path non trovati") from e_error
    status = cv2.imwrite(                       # pylint: disable=E1101
        os.path.join(endpath, file_name), image
    )
    logger.info("ho scritto il file %s in %s come .%s ", file_name, endpath, ext2)
    return status


def unit_masks(file_name, data_path, ext1, ext2, endpath):
    """Normalizza i valori dei pixel delle maschere.
    Permette  di cambiare l'estenzione.

    :type file_name: str
    :param file_path: nome del file della maschera

    :type data_path: str
    :param data_path: percorso della cartella dove si trova la maschera

    :type ext1: str
    :param ext1: stringa identificativa dell'estenzione di partenza della maschera

    :type ext2: str
    :param ext2: stringa identificativa dell'estenzione finale della maschera

    :type endpath: str
    :param endpath: percorso della cartella di arrivo

    :returns: dopo aver salvato il nuovo file, restituisce l'esito e l'immagine
    :rtype: bool, array

    """

    try:
        image = plt.imread(os.path.join(data_path, file_name))
        logger.info("Ho letto %s in %s", file_name, data_path)
    except Exception as e_error:
        raise Exception("immagine o path non trovati!") from e_error

    image = image / 255
    file_name = file_name.replace(f".{ext1}", f".{ext2}")
    status = cv2.imwrite(                       # pylint: disable=E1101
        os.path.join(endpath, file_name), image
    )
    logging.info("ho scritto %s in %s con successo", file_name, endpath)
    return status, image


def read_dataset(        # pylint: disable=R0913
    dataset_path, ext, benign_label, malign_label, x_id="_resized", y_id="_mass_mask"
):
    """Data la cartella con le maschere e le immagini, restituisce i vettori con le immagini,
    le maschere e le classi. Restituisce i vettori come tensori da dare alla rete.

    :type dataset_path: str
    :param dataset_path: Cartella con le immagini e le relative maschere

    :type data_path: str
    :param data_path: percorso della cartella dove si trova la maschera

    :type ext: str
    :param ext: stringa identificativa dell'estenzione delle immagini e maschere

    :type x_id: str
    :param x_id: identificativo delle immagini

    :type x_id: str
    :param x_id: identificativo delle maschere

    :type benign_label: str
    :param benign_label: identificativo delle masse benigne

    :type malign_label: str
    :param malign_label: identificativo delle masse maligne

    :returns: restituisce i vettori con le immagini, le maschere e le classi
    :rtype: array

    """

    fnames = glob.glob(os.path.join(dataset_path, f"*{x_id}.{ext}"))
    logger.info("ho analizzato %s cercando le immagini", dataset_path)
    if fnames == []:
        raise Exception(
            "Niente immagini! Il path è sbagliato, magari x_id o ext sono sbagliati! "
        )

    images_ = []
    masks_ = []
    class_labels = []
    for fname in fnames:
        images_.append(plt.imread(fname)[1:, 1:, np.newaxis])
        masks_.append(plt.imread(fname.replace(x_id, y_id))[1:, 1:, np.newaxis])
        if benign_label in fname:
            class_labels.append(0)
        elif malign_label in fname:
            class_labels.append(1)
    logger.info(
        "ho preso immagini e maschere da %s, ho creato gli array delle immagini, maschere e classi",
        dataset_path,
    )
    return np.array(images_), np.array(masks_), np.array(class_labels)


def read_dataset_big(
    dataset_path_mass, dataset_path_mask, benign_label, malign_label, ext="png", sort=True,
):
    """Versione di read_dataset per il dataset del TCIA.
    Data la cartella con le maschere e le immagini,
    restituisce i vettori con i filepath delle immagini, le maschere e le classi.

    :type dataset_path_mass: str
    :param dataset_path_mass: Cartella con le immagini

    :type dataset_path_mask: str
    :param dataset_path_mask: percorso della cartella dove si trovano le maschera

    :type ext: str
    :param ext: stringa identificativa dell'estenzione delle immagini e maschere


    :returns: restituisce i vettori con i path delle immagini, le maschere e le classi
    :rtype: array



    """

    fnames = glob.glob(os.path.join(dataset_path_mass, f"*.{ext}"))
    logger.info("ho analizzato %s cercando le immagini", dataset_path_mass)

    if fnames == []:
        raise Exception(
            "Niente immagini! Il path è sbagliato, magari ext è sbagliato! "
        )

    masknames = glob.glob(os.path.join(dataset_path_mask, f"*.{ext}"))
    logger.info("ho analizzato %s cercando le maschere", dataset_path_mask)

    if masknames == []:
        raise Exception("Immagini o path non trovati!")
    if sort:
        fnames.sort()
    images_ = []
    masks_ = []
    class_labels = []
    for fname in fnames:
        try:
            assert fname.replace(dataset_path_mass, dataset_path_mask) in masknames
            logger.debug("sto verificando che %s sia in %s", fname, dataset_path_mask)
            masks_.append(fname.replace(dataset_path_mass, dataset_path_mask))
            images_.append(fname)

            if benign_label in fname:
                class_labels.append(0)
            elif malign_label in fname:
                class_labels.append(1)
        except:
            warnings.warn(
                f"{fname} non vi è corrispondenza in {dataset_path_mask},forse immagine mancante"
            )
            logger.warning(
                "%s non vi è corrispondenza in %s, possibile immagine mancante",
                fname,
                dataset_path_mask,
            )

    return np.array(images_), np.array(masks_), np.array(class_labels)


def radiomic_dooer(list_test, endpath, lab, extrc):

    """Funzione per estrarre le feature con pyradiomics e salvarle in un dizionario.

    :type list_test: list
    :param list_test: lista con path immagine e relativa maschera normalizzata
    :type endpath: str
    :param endpath: cartella dove si salva il pickle del dizionario
    :type lab: int
    :param lab: label per indicare la maschera, va da 1 a 255
    :extrc type: str
    :extr param: classe estrattore di pyradiomics

    :returns: dopo aver salvato il pickle, restituisce il tempo impiegato
    :rtype: str


    """

    extr_start = time.perf_counter()
    try:
        logger.debug(
            "sto cercando di estrarre le feature da %s utilizzando come maschera %s",
            list_test[0],
            list_test[1],
        )
        info = extrc.execute(list_test[0], list_test[1], lab)

    except Exception as e_error:
        raise Exception(
            "Pyradiomics: problema di label o path, o non c'è featureextractor"
        ) from e_error

    extr_end = time.perf_counter()
    logging.info("tempo di estrazione: %d", extr_end - extr_start)
    updt_start = time.perf_counter()
    pattern = re.compile(r"[M][\w-]*[0-9]*[\w]{13}")
    logging.debug("in regex il pattern è %s", pattern)
    name = re.findall(pattern, list_test[0])
    logging.debug("il patter che sto cercando è %s", name)

    dict_r = {name[0]: info}
    try:
        with open(os.path.join(endpath, f"feats_{name[0]}.pickle"), "wb") as handle:
            pickle.dump(dict_r, handle, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info("salvato il pickle feats_%s.pickle in %s", name[0], endpath)
    except Exception as e_error:
        raise Exception("Controlla che %s sia giusto"%endpath) from e_error

    del dict_r
    updt_end = time.perf_counter()
    logger.info("time to update:%d", updt_end - updt_start)

    return "time to update:{updt_end-updt_start}"


def read_pgm_as_sitk(image_path):
    """Legge un .pgm come una immagine Simple ITK

    :type image_path: str
    :param image_path: path dell'immagine voluta
    :returns: restituisce l'immagine da far leggere a pyradiomics
    :rtype: array
    """
    try:
        np_array = np.asarray(Image.open(image_path))
    except Exception as e_error:
        raise Exception('impossibile leggere %s'%image_path) from e_error
    logger.info("sto leggendo %s", image_path)
    try:
        sitk_image = sitk.GetImageFromArray(np_array)
    except Exception as e_error:
        raise Exception('impossibile convertire %s in ITK'%image_path) from e_error
    logger.info("sto convertendo %s", image_path)

    return sitk_image


def dict_update_radiomics(data_path, dictionary):

    """Funzione per unire i vari dizionari creati con radiomic_dooer per poi creare il dataframe

    :type data_path: str
    :param data_path: percorso del pickle da aprire
    :type dictionary: dict
    :param dictionary: dizionario generale del dataframe
    :returns: restituisce il dizionario aggiornato
    :rtype: dict


    """

    with open(data_path, "rb") as handle:
        logger.info("ho aperto %s", data_path)
        pic_loaded = pickle.load(handle)
        logger.info("ho caricato %s", handle)
        dictionary.update(pic_loaded)
        logger.info("ho fatto un update a %s", dictionary)

    return dictionary


def blender(img1, img2, weight_1, weight_2):
    """Funzione per sovraimporre due immagini con sfumatura

    :type img1: array numpy
    :param img1: immagine da sovrapporre
    :type img2: array numpy
    :param img2: immagine da svrapporre
    :type weight_1: int or float
    :param weight_1: valore di sfumatura di img1
    :type weight_2: int or float
    :param weight_2: valore di sfumatura di img2
    :returns: restituisce l'immagine sovrapposta
    :rtype: array

    """

    try:
        image = cv2.addWeighted(                        # pylint: disable=E1101
            img1, weight_1, img2, weight_2, 0
        )
        logger.debug(
            "sto cercando di sovrapporre le immagini con pesi rispettivamente %d e %d",
            weight_1,
            weight_2,
        )
    except Exception as e_error:
        raise Exception(
            "Sovrapposizione non riuscita.Immagini o variabili d'ingresso non adeguate."
        ) from e_error

    logger.info("sovrapposizione riuscita")
    return image


def dice(pred, true, k=1):
    """Funzione per calcolare l'indice di Dice

    :type pred: array numpy
    :param pred: immagini predette dal modello
    :type true: array numpy
    :param true: immagini target
    :type k: int
    :param k: valore pixel true della maschera
    :returns: restituisce il valore di Dice
    :rtype: float


    """

    intersection = np.sum(pred[true == k]) * 2.0
    try:
        dice_value = intersection / (np.sum(pred) + np.sum(true))
    except ZeroDivisionError:
        logger.exception("provato a dividere per zero!")
    logger.info("calcolato correttamente il dice ottenendo %d", dice_value)
    return dice_value


def dice_vectorized(pred, true, k=1):
    """
    Versione vettorizzata per calcolare il coefficiente di dice

    :type pred: array numpy
    :param pred: immagini predette dal modello
    :type true: array numpy
    :param true: immagini target
    :type k: int
    :param k: valore pixel true della maschera
    :returns: restituisce il dice medio
    :rtype: float

    """

    intersection = 2.0 * np.sum(pred * (true == k), axis=(1, 2, 3))
    try:
        dice_value = intersection / (
            pred.sum(axis=(1, 2, 3)) + true.sum(axis=(1, 2, 3))
        )
    except ZeroDivisionError:
        logger.exception("provato a dividere per zero!")
    logger.info("calcolato correttamente il dice medio ottenendo %d", dice_value.mean())
    return dice_value


def modelviewer(model):
    """
    Funzione per visualizzare l'andamento della loss di training
     e validazione per l'autoencoder e per il classificatore
    :type model: str
    :param model: history del modello di Keras ottenuto dalla funzione

    """

    plt.figure("modelviewer")
    plt.subplot(2, 1, 1)
    plt.title("autoencoder")
    try:
        plt.plot(model.history["decoder_output_loss"])
        plt.plot(model.history["val_decoder_output_loss"])
    except Exception as e_error:
        raise Exception(
            "Model non ha i campi decoder_output_loss o val_decoder_output_loss"
        ) from e_error

    plt.legend(["loss", "val_loss"])
    plt.subplot(2, 1, 2)
    plt.title("classifier")
    try:
        plt.plot(model.history["classification_output_loss"])
        plt.plot(model.history["val_classification_output_loss"])
    except Exception as e_error:
        raise Exception(
            "Model non ha i campi classification_output_loss o val_classification_output_loss"
        ) from e_error

    plt.legend(["loss", "val_loss"])
    plt.show()


def heatmap(input_image, model,show=True):
    """
    Funzione che mostra la heatmap dell'ultimo layer convoluzionale
    prima del classificatore senza funzionalità radiomiche

    :type input_image: array numpy
    :param input_image: immagine da segmentare

    :type model: str
    :param model: modello allenato
    :returns: Plotta la heatmap sovrapposta e l'immagine, restituisce la heatmap
    :rtype: array


    """

    img_tensor = input_image[np.newaxis, ...]
    preds = model.predict(img_tensor)[1]
    argmax = np.argmax(preds)
    conv_layer = model.get_layer("last_conv")
    heatmap_model = models.Model([model.inputs], [conv_layer.output, model.output])

    with tf.GradientTape() as gtape:
        conv_output, predictions = heatmap_model(img_tensor)
        loss = predictions[1][:, np.argmax(predictions[1])]
        grads = gtape.gradient(loss, conv_output)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

    heat_map = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    heat_map = np.maximum(heat_map, 0)
    max_heat = np.max(heat_map)
    if max_heat == 0:
        max_heat = 1e-10
    heat_map /= max_heat

    if show:
        input_image = np.asarray(255 * input_image, np.uint8)
        heat_map = np.asarray(255 * heat_map.squeeze(), np.uint8)

        heat_map_res = cv2.resize(                      # pylint: disable=E1101
            heat_map, (input_image.shape[1], input_image.shape[0])
        )

        plt.figure("Heatactivation")
        plt.subplot(1,3,1)
        plt.imshow(input_image)
        plt.title('image')
        plt.subplot(1,3,2)
        plt.imshow(heat_map)
        plt.title('heatmap')
        plt.subplot(1,3,3)
        plt.imshow(blender(input_image, heat_map_res, 1, 1))
        if argmax == 1:
            plt.title("the mass is malign")
        else:
            plt.title("the mass is benign")
        plt.show()
    return heat_map


def heatmap_rad(input_image, feature, model,show=True):
    """
    Funzione che mostra la heatmap dell'ultimo layer convoluzionale
    prima del classificatore con funzionalità radiomiche

    :type input_image: array numpy
    :param input_image: immagine da segmentare
    :type feature: array numpy
    :param feature: feature estratte con pyradiomics
    :type model: class
    :param model: modello allenato
    :returns: Plotta la heatmap sovrapposta e l'immagine, restituisce la heatmap
    :rtype: array
    """

    img_tensor = input_image[np.newaxis, ...]
    feature_tensor = feature[np.newaxis, ...]
    preds = model.predict([img_tensor, feature_tensor])[1]
    argmax = np.argmax(preds)
    conv_layer = model.get_layer("last_conv")
    heatmap_model = models.Model([model.inputs], [conv_layer.output, model.output])

    with tf.GradientTape() as gtape:
        conv_output, predictions = heatmap_model([img_tensor, feature_tensor])
        loss = predictions[1][:, np.argmax(predictions[1])]
        grads = gtape.gradient(loss, conv_output)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

    heat_map = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    heat_map = np.maximum(heat_map, 0)
    max_heat = np.max(heat_map)
    if max_heat == 0:
        max_heat = 1e-10
    heat_map /= max_heat
    if show:
        input_image = np.asarray(255 * input_image, np.uint8)
        heat_map = np.asarray(255 * heat_map.squeeze(), np.uint8)

        heat_map_res = cv2.resize(                      # pylint: disable=E1101
            heat_map, (input_image.shape[1], input_image.shape[0])
        )

        plt.figure("Heatactivation")
        plt.subplot(1,3,1)
        plt.imshow(input_image)
        plt.title('image')
        plt.subplot(1,3,2)
        plt.imshow(heat_map)
        plt.title('heatmap')
        plt.subplot(1,3,3)
        plt.imshow(blender(input_image, heat_map_res, 1, 1))
        if argmax == 1:
            plt.title("the mass is malign")
        else:
            plt.title("the mass is benign")
        plt.show()
    return heat_map


def plot_roc_curve(fper, tper, auc):
    """Funzione che fa il plot della curva roc

    :type fper: float
    :param fper: percentuale falsi positivi

    :type tper: float
    :param tper: percentuale veri positivi

    """

    plt.figure("AUC")
    plt.plot(fper, tper, color="orange", label="ROC")
    plt.plot([0, 1], [0, 1], color="darkblue", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve with AUC = %.2f" % auc)
    plt.legend()
    plt.show()


def otsu(image, n_items=2):
    """Funzione che implementa l'algoritmo di Otsu per la segmentazione

    :type image: numpy array
    :param fper: immagine da segmentare
    :type n_items: int
    :param n_items: numero di oggetti da segmentare nell'immagine
    :returns: restituisce l'immagine binarizzata
    :rtype: array

    """

    thresholds = threshold_multiotsu(image, classes=n_items)
    regions = np.digitize(image, bins=thresholds)
    return regions
