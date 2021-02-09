"""docstring"""
import argparse
import concurrent.futures
import logging
import os
import re
import time
import warnings
from functools import partial

from PIL import Image
from radiomics import featureextractor

from functioncae import caehelper

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

file_handler = logging.FileHandler("Feature_extraction.log")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


def resizer(list_, endpath, pattern):
    """Funzione per fare il reshape delle maschere
    in maniera che combacino con le immagini a cui sono riferite

    type list_: lista
    param list_: lista che contiene il path della immagine e il path della maschera

    type endpath: stringa
    param endpath: path di arrivo per le nuove maschere

    type pattern: stringa
    param pattern: pattern da trovare per dare il nome corretto alla nuova maschera. Usa regex
    """
    logger.info(
        "cerco di leggere %s e %s da salvare in %s con pattern %s",
        list_[0],
        list_[1],
        endpath,
        pattern,
    )
    start_time = time.perf_counter()
    try:
        image = Image.open(list_[0])
    except:  # pylint: disable=W0702
        warnings.warn("Immagine %s mancante o corrotta", list_[0])
        logger.exception(
            "Immagine %s mancante o corrotta, non riesco a leggerla", list_[0]
        )

    try:
        mask = Image.open(list_[1])
    except:  # pylint: disable=W0702
        warnings.warn("Immagine %s mancante o corrotta", list_[1])
        logger.exception(
            "maschera %s mancante o corrotta, non riesco a leggerla", list_[1]
        )

    try:

        mask = mask.resize(image.size)
        logger.debug(  # pylint: disable=W1203
            f"ho fatto il resize di {list_[1]} usando come dimensione {image.shape} di {list_[0]}"
        )
    except:  # pylint: disable=W0702
        warnings.warn(
            "Non riesco a fare il resize. Sto salvando l'immagine senza fare resize!!!"
        )

        logger.critical(
            "Non riesco a fare il resize. Sto salvando l'immagine senza fare resize!!!"
        )
    try:
        match = re.findall(pattern, list_[0])[0]
        logger.debug("il match del pattern è %s", match)
        filename = os.path.join(endpath, match + ".png")
        mask.save(filename)
        logger.info("salvata la nuova maschera in %s", filename)

    except:  # pylint: disable=W0702
        warnings.warn("Non possibile andare avanti")
        logger.warning(
            "Non possibile andare avanti, non trovo il pattern o non riesco a salvare il file"
        )
    end_time = time.perf_counter()

    logger.info("time elapsed: %d", end_time - start_time)
    return filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estrae le feature con pyradiomics e crea una cartella di maschere di dimensione uguale alle masse " #pylint:disable=C0301
    )
    parser.add_argument(
        "-csv",
        "--csvpath",
        metavar="",
        help="percorso della cartella dove si salva il dataframe",
        default="",
    )
    requiredNamed = parser.add_argument_group("Parametri obbligatori")
    requiredNamed.add_argument(
        "-mp",
        "--mainpath",
        metavar="",
        help="percorso dove si trova la cartella con i dataset generati con dycomdatagen.py",
        required=True,
    )
    args = parser.parse_args()

    DATAPATH = args.mainpath
    MASS_TRAIN = "Train_data"
    MASK_TRAIN = "Train_data_masks"
    MASK_TRAIN_RES = "resized_masks"
    BENIGN_LABEL = "BENIGN"
    MALIGN_LABEL = "MALIGNANT"
    FEATURES_PATH = "feats"

    PATH_MASS_TR = os.path.join(DATAPATH, MASS_TRAIN)
    PATH_MASK_TR = os.path.join(DATAPATH, MASK_TRAIN)

    PATH_MASK_RESIZED = os.path.join(DATAPATH, MASK_TRAIN_RES)

    images_big_train, masks_big_train, class_big_train = caehelper.read_dataset_big(
        PATH_MASS_TR, PATH_MASK_TR, BENIGN_LABEL, MALIGN_LABEL
    )

    if not os.path.exists(PATH_MASK_RESIZED):
        os.makedirs(PATH_MASK_RESIZED)
        logger.info("creato il path %s", PATH_MASK_RESIZED)

    ENDPATH_TR = os.path.join(DATAPATH, FEATURES_PATH)
    if not os.path.exists(ENDPATH_TR):
        os.makedirs(ENDPATH_TR)
        logger.info("creato il path %s", ENDPATH_TR)

    extractor = featureextractor.RadiomicsFeatureExtractor()
    logger.info("inizializzato estrattore di pyradiomics")
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("gldm")
    extractor.enableFeatureClassByName("glcm")
    extractor.enableFeatureClassByName("shape2D")
    extractor.enableFeatureClassByName("firstorder")
    extractor.enableFeatureClassByName("glrlm")
    extractor.enableFeatureClassByName("glszm")
    extractor.enableFeatureClassByName("ngtdm")

    images_masks = [
        [images_big_train[i], masks_big_train[i]]
        for i, _ in enumerate(images_big_train)
    ]

    resize_mt = partial(
        resizer,
        endpath=PATH_MASK_RESIZED,
        pattern=re.compile(r"[M][\w-]*[0-9]*[\w]{13}"),
    )
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor() as executor:

        results = executor.map(resize_mt, images_masks)
        logger.debug("%s", results)
        print(results)
    end = time.perf_counter()

    logger.info("Elapsed time for MT: %d", end - start)

    images_big_train, masks_big_train, class_big_train = caehelper.read_dataset_big(
        PATH_MASS_TR, PATH_MASK_RESIZED, BENIGN_LABEL, MALIGN_LABEL
    )

    images_masks = [
        [images_big_train[i], masks_big_train[i]]
        for i, _ in enumerate(images_big_train)
    ]

    # for item in images_masks:
    #    try:
    #        NAME = caehelper.radiomic_dooer(
    #            item, ENDPATH_TR, 255, extractor
    #        )
    #    except: #pylint: disable=W0702
    #        logger.debug("un file errato: %s",PATH_MASS_TR)

    rad_dooer = partial(
        caehelper.radiomic_dooer, endpath=ENDPATH_TR, lab=255, extrc=extractor
    )

    start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor() as executor:

        results = executor.map(rad_dooer, images_masks)
        logger.debug("%s", results)
        print(results)
    end = time.perf_counter()

    logger.info("Elapsed time for MT: %d", end - start)

    new_dict = {}
    new_dict_up = partial(caehelper.dict_update_radiomics, dictionary=new_dict)
    new_list = []
    list_items = next(os.walk(ENDPATH_TR))[2]
    for items in list_items:
        new_list.append(os.path.join(ENDPATH_TR, items))

    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor() as executor:

        results = executor.map(new_dict_up, new_list)
        logger.debug("%s", results)
        print(results)
    end = time.perf_counter()

    logger.info("Elapsed time for MT: %d", end - start)

    import pandas as pd

    Pandata_big = pd.DataFrame(new_dict)

    for i, _ in enumerate(Pandata_big.index):
        if "diagnostics" in Pandata_big.index[i]:
            print(i)
        else:
            pass

    Pandatabigframe = Pandata_big.drop(Pandata_big.index[0:22]).T

    gfg_csv_data = Pandatabigframe.to_csv(
        os.path.join(args.csvpath, "Bigframe_test.csv"), index=True
    )
