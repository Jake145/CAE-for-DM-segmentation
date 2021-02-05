"""docstring"""
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


def dycomrconverter(data, dir_mass, dir_mask, end_path_mass, end_path_mask):

    """Funzione per leggere il dataframe e salvare l'immagine e la relativa maschera in .png da DICOM

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
        pattern = re.compile(r"[\w-]+[\w]\b")
        match = re.findall(pattern, d_c)[0]
        logger.debug("trovato come nome della immagine: %s in %s", match, _dir)
        dir_ = os.path.join(_dir, match)
        logger.info("impostato come directory di arrivo per le immagini:%s", dir_)
        return dir_

    def dicomsaver(dcm, d_c, file_endpath):
        pattern = re.compile(r"[\w-]+[\w]\b")
        match = re.findall(pattern, d_c)[0]
        image = dcm.pixel_array
        pathology = data["pathology"]
        apply_voi_lut(image, dcm)
        filename = f"{match}_{pathology}.png"
        filename = os.path.join(file_endpath, filename)
        logger.debug("salvato in %s", filename)
        status = cv2.imwrite(filename, image)
        logger.info("correttamente salvato %s", filename)
        return status

    dir_masses = dir_reader(dir_mass, data["image file path"])
    files = search_files(dir_masses, ".dcm")
    logger.info("sto cercando i dycom in %s", dir_masses)
    try:
        assert len(files) == 1
        logger.debug("ho verificato che nella cartella vi è una sola immagine")
        # questo va bene nel nostro caso specifico. Nel caso generale sarebbe meglio guardare il nome del file dal dycom, ma questo modo è più rapido
    except:
        raise Exception(
            "Nella directory %s delle singole immagini ci sono più file", dir_masses
        )

    f = files[0]
    logger.info("ho letto %s", f)
    d_s = pydicom.read_file(f)
    logger.info("ho letto correttamente il dicom di %s", f)

    status_mass = dicomsaver(d_s, data["image file path"], end_path_mass)

    # ora facciamo la maschera
    dir_masks = dir_reader(dir_mask, ["ROI mask file path"])
    files = search_files(dir_masks, ".dcm")
    logger.info("sto cercando i dicom in %s", dir_masks)

    try:
        assert len(files) <= 2
        logger.debug("ho verificato che nella cartella vi è al più due immagini")

    except:
        raise Exception(
            "Nella directory %s delle maschere ci sono più masks", dir_masks
        )

    for f in files:

        d_g = pydicom.dcmread(f)
        logger.debug("ho letto correttamente il dicom di %s", f)
        good = d_g.SeriesDescription
        if good == "ROI mask images":

            d_s_mask = pydicom.read_file(f)
            logger.debug("in %s vi è una maschera", f)
        elif good == "cropped images":
            logger.debug("in %s vi è una ROI", f)

        else:
            raise Exception(f"{f} File sconosciuto!")

    status_mask = dicomsaver(d_s_mask, data["ROI mask file path"], end_path_mask)

    return status_mass, status_mask


###

if __name__ == "__main__":

    # Definiamo i path e il dataframe

    DATAPATH_MASK_TRAIN = "E:/massdata"  # modificare con il proprio path
    DATAPATH_MASS_TRAIN = "E:/massdatafull"  # modificare con il proprio path
    MAIN_DIRECTORY = "CBIS-DDSM"

    CSV_PATH = "C:/Users/pensa/Desktop/CAE/csvs"
    CSV_FILENAME = "mass_case_description_train_set.csv"
    df = pd.read_csv(os.path.join(CSV_PATH, CSV_FILENAME))

    datas = []
    for i in range(len(df)):
        datas.append(df.iloc[i, :])

    DIRECTORY_MASS = os.path.join(DATAPATH_MASS_TRAIN, MAIN_DIRECTORY)
    DIRECTORY_MASK = os.path.join(DATAPATH_MASK_TRAIN, MAIN_DIRECTORY)

    ENDPATH = "E:/Mass_data_new"
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

    ##multi thread
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor() as executor:

        results = executor.map(dycomrconverter_new, datas)
        print(results)
    end = time.perf_counter()

    logger.info("Elapsed time for MT:%d", end - start)
