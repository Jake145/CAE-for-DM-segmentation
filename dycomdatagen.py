"""docstring"""
import argparse
import concurrent.futures
import logging
import os
import re
import time
from functools import partial
import cv2
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

file_handler = logging.FileHandler("Dicomdatagen.log")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


def dycomrconverter(             # pylint: disable=R0915,R0914
    data, dir_mass, dir_mask, end_path_mass, end_path_mask
):

    """Funzione per leggere il dataframe e salvare l'immagine
    e la relativa maschera in .png da DICOM

    :type data: dataframe.iloc
    :param data: riga del dataframe corrispondente alla maschera
    :type dir_mass: str
    :param dir_mass: path dove si trovano le immagini
    :type dir_mask: str
    :param dir_mask: path dove si trovano le maschere
    :type end_path_mass: str
    :param end_path_mass: path di arrivo per le immagini
    :type end_path_mask: str
    :param end_path_mask: path di arrivo per le maschere
    :returns: restituisce lo stato del salvataggio delle immagini
    :rtype: bool,bool


    """
    logger.debug(
        "Per estrarre le immagini e maschere, carico da %s e %s con cartelle di arrivo %s e %s",
        dir_mass,
        dir_mask,
        end_path_mass,
        end_path_mask,
    )

    def search_files(directory=".", extension=""):
        """Funzione per cercare ogni file dentro una certa cartella,
        anche se contiene sottocartelle

        :type directory: str
        :param directory: cartella in cui si vuole cercare i file
        :type extension: str
        :param extension: estensione dei file, di default sono tutti
        :returns: lista con i path dei file
        :rtype:list
        """
        extension = extension.lower()
        filelist = []
        for dirpath, _, files in os.walk(directory):
            for name in files:
                if extension and name.lower().endswith(extension):
                    filelist.append(os.path.join(dirpath, name))
                elif not extension:
                    filelist.append(os.path.join(dirpath, name))
        return filelist

    def dir_reader(_dir, d_c):
        """data la directory principale, trova la directory con il nome
        corrispondente al dicom

        :type _dir: str
        :param _dir: path principale
        :type d_c: dataframe.iloc
        :param d_c: riga del dataframe corrispondente al dicom
        :returns: directory del dicom
        :rtype:str
        """
        pattern = re.compile(r"[\w-]+[\w]\b")
        match = re.findall(pattern, d_c)[0]
        logger.debug("trovato come nome della immagine: %s in %s", match, _dir)
        dir_ = os.path.join(_dir, match)
        logger.info("impostato come directory di arrivo per le immagini:%s", dir_)
        return dir_

    def dicomsaver(dcm, d_c, file_endpath):
        """Salva il dicom in .png con  nome che riflette le caratteristiche
        date dalla riga del dataframe
        :type dcm: dicom
        :param dcm: immagine in formato .dcm
        :type d_c: dataframe.iloc
        :param d_c: riga del dataframe corrispondente al dicom
        :type file_endpath: str
        :param file_endpath: cartella dove viene salvata l'immagine
        :returns: stato del salvataggio
        :rtype:bool
        """
        pattern = re.compile(r"[\w-]+[\w]\b")
        match = re.findall(pattern, d_c)[0]
        image = dcm.pixel_array
        pathology = data["pathology"]
        view=data["abnormality id"]
        apply_voi_lut(image, dcm)
        filename = f"{match}_{pathology}_{view}.png"
        filename = os.path.join(file_endpath, filename)
        logger.debug("salvato in %s", filename)
        status = cv2.imwrite(filename, image) #pylint:disable=E1101
        logger.info("correttamente salvato %s", filename)
        return status

    dir_masses = dir_reader(dir_mass, data["image file path"])
    files = search_files(dir_masses, ".dcm")
    logger.info("sto cercando i dycom in %s", dir_masses)
    try:
        assert len(files) == 1
        logger.debug("ho verificato che nella cartella vi è una sola immagine")
    except Exception as e_error:
        raise Exception(
            "Nella directory {} delle singole immagini ci sono più file".format(dir_masses)
        ) from e_error

    f_images = files[0]
    logger.info("ho letto %s", f_images)
    d_s = pydicom.read_file(f_images)
    logger.info("ho letto correttamente il dicom di %s", f_images)

    status_mass = dicomsaver(d_s, data["image file path"], end_path_mass)

    # ora facciamo la maschera
    dir_masks = dir_reader(dir_mask, data["ROI mask file path"])
    files = search_files(dir_masks, ".dcm")
    logger.info("sto cercando i dicom in %s", dir_masks)
    try:
        assert len(files) > 0
    except Exception as e_error:
        raise Exception(
            "Nella directory {} delle maschere non ci sono file".format(dir_masks)
        ) from e_error

    try:
        assert len(files) <= 2
        # delle volte nel TCIA mettono le roi e le maschere in cartelle separate
        logger.debug("ho verificato che nella cartella vi sono al più due immagini")

    except Exception as e_error:
        raise Exception(
            "Nella directory {} delle maschere ci sono più masks".format(dir_masks)
        ) from e_error

    for f_masks in files:

        d_g = pydicom.dcmread(f_masks)
        logger.debug("ho letto correttamente il dicom di %s", f_masks)
        good = d_g.SeriesDescription
        if good == "ROI mask images":

            d_s_mask = pydicom.read_file(f_masks)
            logger.debug("in %s vi è una maschera", f_masks)
            dicomsaver(d_s_mask, data["image file path"], end_path_mask)

        elif good == "cropped images":
            logger.debug("in %s vi è una ROI", f_masks)

        else:
            raise Exception(f"{f_masks} File sconosciuto!")

    return status_mass




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Legge i DICOM CBIS-DDSM scaricati dal TCIA. "
    )
    parser.add_argument(
        "-csv",
        "--csvpath",
        metavar="",
        help="percorso della cartella dove si trovano i csv",
        default="",
    )
    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        help="si usa il dataset di testing",
    )
    requiredNamed = parser.add_argument_group("Parametri obbligatori")
    requiredNamed.add_argument(
        "-mp",
        "--masspath",
        metavar="",
        help="percorso della cartella dove si trovano i dicom delle masse",
        required=True,
    )
    requiredNamed.add_argument(
        "-msk",
        "--maskpath",
        metavar="",
        help="percorso della cartella dove si trovano i dicom delle maschere",
        required=True,
    )
    requiredNamed.add_argument(
        "-ep",
        "--endpath",
        metavar="",
        help="percorso della cartella dove viene salvato il nuovo dataset",
        required=True,
    )
    args = parser.parse_args()

    # Definiamo i path e il dataframe

    DATAPATH_MASK_TRAIN = args.maskpath
    DATAPATH_MASS_TRAIN = args.masspath
    MAIN_DIRECTORY = "CBIS-DDSM"

    CSV_PATH = args.csvpath
    if args.test:
        CSV_FILENAME = "mass_case_description_test_set.csv"
    else:
        CSV_FILENAME = "mass_case_description_train_set.csv"

    df = pd.read_csv(os.path.join(CSV_PATH, CSV_FILENAME))

    datas = []
    for i in range(len(df)):
        datas.append(df.iloc[i, :])

    DIRECTORY_MASS = os.path.join(DATAPATH_MASS_TRAIN, MAIN_DIRECTORY)
    DIRECTORY_MASK = os.path.join(DATAPATH_MASK_TRAIN, MAIN_DIRECTORY)

    ENDPATH = args.endpath

    if not os.path.exists(ENDPATH):
        os.makedirs(ENDPATH)
    logger.info("creato il path %s", ENDPATH)

    TRPATH = "Train_data"
    ENDPATH_1 = os.path.join(ENDPATH, TRPATH)
    if not os.path.exists(ENDPATH_1):
        os.makedirs(ENDPATH_1)
    logger.info("creato il path %s", ENDPATH_1)

    TRPATH_MASK = "Train_data_masks"
    ENDPATH_2 = os.path.join(ENDPATH, TRPATH_MASK)
    if not os.path.exists(ENDPATH_2):
        os.makedirs(ENDPATH_2)
    logger.info("creato il path %s", ENDPATH_2)

    dycomrconverter_new = partial(
        dycomrconverter,
        dir_mass=DIRECTORY_MASS,
        dir_mask=DIRECTORY_MASK,
        end_path_mass=ENDPATH_1,
        end_path_mask=ENDPATH_2,
    )

    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor() as executor:

        results = executor.map(dycomrconverter_new, datas)
        print(results)
    end = time.perf_counter()

    logger.info("Elapsed time for MT:%.2f", end - start)
