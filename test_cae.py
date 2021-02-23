"""Unit tests"""
import logging
import os
import pickle
import tempfile
import unittest

import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from radiomics import featureextractor
from skimage.io import imread

# import dycomdatagen
import feature_extraction
from functioncae import caehelper, classes_cae

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")

file_handler = logging.FileHandler("Unittest.log")
file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


class TestCAE(unittest.TestCase):  # pylint:disable=R0902
    """Unit test class"""

    @classmethod
    def setUpClass(cls):  # pylint: disable=R0915
        """Metodo di set up per tutti i test fatto all'inizio"""

        cls.shape_im = (124, 124)
        cls.image_ones = np.ones(cls.shape_im)
        cls.image_zeros = np.zeros(cls.shape_im)
        cls.image_square = np.zeros(cls.shape_im)
        cls.image_square[50:100, 50:100] = 1
        cls.image_mask = cls.image_square * 255

        cls.temp_dir = tempfile.TemporaryDirectory()

        cls.filename1 = "Image1_benign_resized.png"
        cls.path_1 = os.path.join(cls.temp_dir.name, cls.filename1)
        cls.new_img_1 = Image.fromarray(cls.image_ones)
        cls.new_img_1 = cls.new_img_1.convert("L")
        cls.new_img_1.save(cls.path_1)

        cls.filename2 = "Image2_malign_resized.png"
        cls.path_2 = os.path.join(cls.temp_dir.name, cls.filename2)
        cls.new_img_2 = Image.fromarray(cls.image_zeros)
        cls.new_img_2 = cls.new_img_2.convert("L")
        cls.new_img_2.save(cls.path_2)

        cls.filename3 = "Image2_malign_mass_mask.png"
        cls.path_3 = os.path.join(cls.temp_dir.name, cls.filename3)
        cls.new_img_3 = Image.fromarray(cls.image_square)
        cls.new_img_3 = cls.new_img_3.convert("L")
        cls.new_img_3.save(cls.path_3)

        cls.filename4 = "Image1_benign_mass_mask.png"
        cls.path_4 = os.path.join(cls.temp_dir.name, cls.filename4)
        cls.new_img_4 = Image.fromarray(cls.image_mask)
        cls.new_img_4 = cls.new_img_4.convert("L")
        cls.new_img_4.save(cls.path_4)

        cls.filename5 = "Only_Ones.pgm"
        cls.path_5 = os.path.join(cls.temp_dir.name, cls.filename5)
        cls.new_img_5 = Image.fromarray(cls.image_ones)
        cls.new_img_5 = cls.new_img_5.convert("L")
        cls.new_img_5.save(cls.path_5)

        # setup per le immagini grandi

        cls.shape_im_big = (5000, 3000)
        cls.image_ones_big = np.ones(cls.shape_im_big)
        cls.image_zeros_big = np.zeros(cls.shape_im_big)
        cls.image_square_big = np.zeros(cls.shape_im_big)
        cls.image_square_big[500:1000, 500:1000] = 1
        cls.image_mask_big = cls.image_square_big * 255

        cls.temp_dir_big_mass = tempfile.TemporaryDirectory()

        cls.filename1_big = "Image1_benign.png"
        cls.path_1_big = os.path.join(cls.temp_dir_big_mass.name, cls.filename1_big)
        cls.new_img_1_big = Image.fromarray(cls.image_ones_big)
        cls.new_img_1_big = cls.new_img_1_big.convert("L")
        cls.new_img_1_big.save(cls.path_1_big)

        cls.filename2_big = "Image2_malign.png"
        cls.path_2_big = os.path.join(cls.temp_dir_big_mass.name, cls.filename2_big)
        cls.new_img_2_big = Image.fromarray(cls.image_zeros_big)
        cls.new_img_2_big = cls.new_img_2_big.convert("L")
        cls.new_img_2_big.save(cls.path_2_big)

        cls.temp_dir_big_masks = tempfile.TemporaryDirectory()

        cls.filename3_big = "Image1_benign.png"
        cls.path_3_big = os.path.join(cls.temp_dir_big_masks.name, cls.filename3_big)
        cls.new_img_3_big = Image.fromarray(cls.image_square_big)
        cls.new_img_3_big = cls.new_img_3_big.convert("L")
        cls.new_img_3_big.save(cls.path_3_big)

        cls.filename4_big = "Image2_malign.png"
        cls.path_4_big = os.path.join(cls.temp_dir_big_masks.name, cls.filename4_big)
        cls.new_img_4_big = Image.fromarray(cls.image_mask_big)
        cls.new_img_4_big = cls.new_img_4_big.convert("L")
        cls.new_img_4_big.save(cls.path_4_big)

        cls.list_test = [cls.path_1, cls.path_3_big]
        cls.list_test_wrong = ["wrongpath", "wrongpathmask"]

        # cartella pickle
        cls.pickles = tempfile.TemporaryDirectory()
        cls.pickle = {"thispickle": "thatpickle"}
        cls.pickle_filename = "Pick.pickle"
        cls.pickle_path = os.path.join(cls.pickles.name, cls.pickle_filename)
        with open(cls.pickle_path, "wb") as handle:
            pickle.dump(cls.pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)

        cls.filename_rad1 = "Mass-Training_P_00001_LEFT_CC_MALIGNANT.png"
        cls.path_rad1 = os.path.join(cls.pickles.name, cls.filename_rad1)
        cls.new_img_rad1 = Image.fromarray(cls.image_square)
        cls.new_img_rad1 = cls.new_img_rad1.convert("L")
        cls.new_img_rad1.save(cls.path_rad1)

        cls.filename_rad2 = "Mask-Training_P_00001_LEFT_CC_MALIGNANT.png"
        cls.path_rad2 = os.path.join(cls.pickles.name, cls.filename_rad2)
        cls.new_img_rad2 = Image.fromarray(cls.image_mask)
        cls.new_img_rad2 = cls.new_img_rad2.convert("L")
        cls.new_img_rad2.save(cls.path_rad2)

        cls.list_ = [cls.path_rad1, cls.path_rad2]

        cls.extractor = featureextractor.RadiomicsFeatureExtractor()
        cls.extractor = featureextractor.RadiomicsFeatureExtractor()
        cls.extractor.disableAllFeatures()
        cls.extractor.enableFeatureClassByName("gldm")
        cls.extractor.enableFeatureClassByName("glcm")
        cls.extractor.enableFeatureClassByName("shape2D")
        cls.extractor.enableFeatureClassByName("firstorder")
        cls.extractor.enableFeatureClassByName("glrlm")
        cls.extractor.enableFeatureClassByName("glszm")
        cls.extractor.enableFeatureClassByName("ngtdm")

    @classmethod
    def tearDownClass(cls):
        """Metodo di tear down per tutti i test fatto all'inizio"""
        cls.temp_dir.cleanup()
        cls.temp_dir_big_mass.cleanup()
        cls.temp_dir_big_masks.cleanup()
        cls.pickles.cleanup()

    def setUp(self):
        """Metodo di set up per tutti i test fatto all'inizio di ogni singolo test"""
        self.endpath = tempfile.TemporaryDirectory()

    def tearDown(self):
        """Metodo di tear down per tutti i test fatto all'inizio di ogni singolo test"""

        self.endpath.cleanup()

    def test_save_newext(self):
        """test per la funzione che salva un'immagine in una nuova estensione"""

        stat = caehelper.save_newext(
            self.filename5, self.temp_dir.name, "pgm", "png", self.endpath.name
        )
        self.assertTrue(stat)

        with self.assertRaises(Exception):
            caehelper.save_newext(
                "failfile.pgm", self.temp_dir.name, "pgm", "png", self.endpath.name
            )

    def test_unit_mask(self):
        """test per la funzione che normalizza una maschera/immagine"""
        stat, image = caehelper.unit_masks(
            self.filename4, self.temp_dir.name, "png", "pgm", self.endpath.name
        )
        self.assertTrue(stat)
        np.testing.assert_allclose(np.round(image), self.image_zeros)
        with self.assertRaises(Exception):
            caehelper.unit_masks(
                "failfile.pgm", self.temp_dir.name, "pgm", "png", self.endpath.name
            )

    def test_read_pgm(self):
        """test per la funzione che legge un pgm in ITK"""
        image_ = caehelper.read_pgm_as_sitk(self.path_5)
        np.testing.assert_array_equal(
            image_, self.image_ones.reshape(self.shape_im[0] ** 2)
        )
        with self.assertRaises(Exception):
            caehelper.read_pgm_as_sitk("wrongpath.pgm")

    def test_read_dataset(self):
        """test per la funzione che legge un dataset e restituisce
        i vettori delle immagini, delle maschere e delle classi"""
        images_, masks_, class_ = caehelper.read_dataset(
            self.temp_dir.name,
            "png",
            "benign",
            "malign",
            x_id="_resized",
            y_id="_mass_mask",
        )
        self.assertEqual(len(images_), len(masks_))
        self.assertEqual(len(images_), len(class_))
        np.testing.assert_array_equal(class_, np.array([0, 1]))
        self.assertNotEqual(list(class_), [0, 0])
        self.assertNotEqual(list(class_), [1, 1])

        with self.assertRaises(Exception):
            images_, masks_, class_ = caehelper.read_dataset(
                self.temp_dir.name,
                "jpg",
                "benign",
                "malign",
                x_id="_resized",
                y_id="_mass_mask",
            )
        with self.assertRaises(Exception):
            images_, masks_, class_ = caehelper.read_dataset(
                "C:/Mock/fakedirpath",
                "png",
                "benign",
                "malign",
                x_id="_resized",
                y_id="_mass_mask",
            )
        with self.assertRaises(Exception):
            images_, masks_, class_ = caehelper.read_dataset(
                self.temp_dir.name,
                "png",
                "benign",
                "malign",
                x_id="_ReSiz3D",
                y_id="_mess_mansk",
            )

    def test_read_dataset_big(self):
        """test per la funzione che legge un dataset e restituisce
        i vettori dei path delle immagini e delle maschere e le classi
        """
        images_, masks_, class_ = caehelper.read_dataset_big(
            self.temp_dir_big_mass.name,
            self.temp_dir_big_masks.name,
            "benign",
            "malign",
            ext="png",
        )
        self.assertEqual(len(images_), len(masks_))
        self.assertEqual(len(images_), len(class_))
        np.testing.assert_array_equal(class_, np.array([0, 1]))
        self.assertNotEqual(list(class_), [0, 0])
        self.assertNotEqual(list(class_), [1, 1])

        with self.assertRaises(Exception):
            images_, masks_, class_ = caehelper.read_dataset_big(
                self.temp_dir_big_mass.name,
                self.temp_dir_big_masks.name,
                "benign",
                "malign",
                ext="jpg",
            )
        with self.assertRaises(Exception):
            images_, masks_, class_ = caehelper.read_dataset_big(
                "C:/Mock/fakedirpath",
                "C:/Mock/fakedirpath2",
                "benign",
                "malign",
                ext="png",
            )

    def test_dict_update_radiomics(self):
        """test per la funzione che aggiorna un dizionario esistente leggendo un file .pickle"""
        mockdict = {}
        diz = caehelper.dict_update_radiomics(self.pickle_path, mockdict)
        self.assertEqual(len(diz), 1)

    def test_blender(self):
        """test per la funzione che sovrappone due immagini"""
        image = caehelper.blender(self.image_ones, self.image_ones, 1, 1)
        np.testing.assert_array_equal(image, np.full(self.shape_im, 2))
        with self.assertRaises(Exception):
            caehelper.blender("fake.png", "string.txt", "really wrong", "oh noes")

    def test_dice(self):
        """test per il calcolo del dice per singola immagine"""
        dix = caehelper.dice(self.image_ones, self.image_square)
        dix_1 = caehelper.dice(self.image_ones, self.image_ones)
        dix_0 = caehelper.dice(self.image_zeros, self.image_ones)
        self.assertEqual(np.round(dix, 2), 0.28)
        self.assertEqual(np.round(dix_1), 1)
        self.assertEqual(np.round(dix_0), 0)

    def test_modelviewer(self):
        """test per la funzione che plotta le loss del modello appena allenato"""
        with self.assertRaises(Exception):
            caehelper.modelviewer("nothing")

    def test_otsu(self):
        """test per la funzione che fa la segmentazione di otsu"""
        masked = caehelper.otsu(0.5 * self.image_square)
        np.testing.assert_array_equal(masked, self.image_square)

    def test_masses_classes(self):
        """test per la classe che restituisce i vettori dei dati
        in ingresso in batch con data augmentation
        """
        train_datagen = ImageDataGenerator(horizontal_flip=True, fill_mode="reflect")
        transform = train_datagen.get_random_transform(self.shape_im)
        feats = [1, 2, 3]

        images_, masks_, class_ = (
            np.array([self.image_ones, self.image_zeros]),
            np.array([self.image_mask, self.image_square]),
            np.array([0, 1]),
        )

        images_big, masks_big, class_big = (
            np.array([self.path_1_big, self.path_2_big]),
            np.array([self.path_3_big, self.path_4_big]),
            np.array([0, 1]),
        )

        gen1 = classes_cae.MassesSequence(
            images_, masks_, class_, train_datagen, batch_size=1, shape=self.shape_im
        )
        np.testing.assert_array_equal(gen1.images, images_)
        np.testing.assert_array_equal(gen1.masks, masks_)
        np.testing.assert_array_equal(gen1.label_array, class_)
        np.testing.assert_array_equal(gen1.shape, self.shape_im)
        np.testing.assert_array_equal(gen1.process(images_[0], transform), images_[0])
        self.assertEqual(gen1.batch_size, 1)
        self.assertEqual(len(gen1), 2)

        gen2 = classes_cae.MassesSequenceRadiomics(
            images_,
            masks_,
            class_,
            feats,
            train_datagen,
            batch_size=1,
            shape=self.shape_im,
        )
        np.testing.assert_array_equal(gen2.images, images_)
        np.testing.assert_array_equal(gen2.masks, masks_)
        np.testing.assert_array_equal(gen2.label_array, class_)
        np.testing.assert_array_equal(gen2.shape, self.shape_im)
        np.testing.assert_array_equal(gen2.features, feats)

        np.testing.assert_array_equal(gen2.process(images_[0], transform), images_[0])
        self.assertEqual(gen2.batch_size, 1)
        self.assertEqual(len(gen2), 2)

        gen3 = classes_cae.MassesSequenceRadiomicsBig(
            images_big,
            masks_big,
            class_big,
            feats,
            train_datagen,
            batch_size=1,
            shape=(2048, 1536),
            shape_tensor=(2048, 1536, 1),
        )

        np.testing.assert_array_equal(gen3.images, images_big)
        np.testing.assert_array_equal(gen3.masks, masks_big)
        np.testing.assert_array_equal(gen3.label_array, class_big)
        np.testing.assert_array_equal(gen3.shape, (2048, 1536))
        np.testing.assert_array_equal(gen3.shape_tensor, (2048, 1536, 1))
        np.testing.assert_array_equal(gen3.features, feats)

        np.testing.assert_array_equal(
            gen3.process(imread(images_big[0]), transform), imread(images_big[0])
        )
        self.assertEqual(gen3.batch_size, 1)
        self.assertEqual(len(gen3), 2)

        gen_val = classes_cae.ValidatorGenerator(
            images_big,
            masks_big,
            class_big,
            feats,
            batch_size=1,
            shape=(2048, 1536),
            shape_tensor=(2048, 1536, 1),
        )

        np.testing.assert_array_equal(gen_val.images, images_big)
        np.testing.assert_array_equal(gen_val.masks, masks_big)
        np.testing.assert_array_equal(gen_val.label_array, class_big)
        np.testing.assert_array_equal(gen_val.shape, (2048, 1536))
        np.testing.assert_array_equal(gen_val.shape_tensor, (2048, 1536, 1))
        np.testing.assert_array_equal(gen_val.features, feats)

        self.assertEqual(gen_val.batch_size, 1)
        self.assertEqual(len(gen_val), 2)

    def test_radiomic(self):
        """test per l'estrazione di feature radiomiche"""

        self.list_ = [self.path_rad1, self.path_rad2]
        caehelper.radiomic_dooer(self.list_, self.endpath.name, 255, self.extractor)
        object_ = pd.read_pickle(
            os.path.join(
                self.endpath.name,
                "feats_Mass-Training_P_00001_LEFT_CC_MALIGNANT.pickle",
            )
        )
        self.assertEqual(len(object_), 1)
        with self.assertRaises(Exception):
            caehelper.radiomic_dooer(self.list_, self.endpath.name, 1, self.extractor)
        with self.assertRaises(Exception):
            caehelper.radiomic_dooer(self.list_, "fakepath", 255, self.extractor)
        with self.assertRaises(Exception):
            caehelper.radiomic_dooer(
                ["wrong.png", "wrongmask.png"], self.endpath.name, 255, self.extractor
            )

    def test_resizer(self):
        """test per la funzione parallelizzabile che salva la maschera normalizzata
        corrispondente a una immagine (Multi-Threading)
        """
        self.list_test = [self.path_1, self.path_3_big]
        fname = feature_extraction.resizer(
            self.list_test, self.endpath.name, "Image1_benign"
        )
        np.testing.assert_array_equal(imread(fname).shape, self.image_ones.shape)
        with self.assertRaises(Exception):
            fname = feature_extraction.resizer(
                self.list_test, self.endpath.name, "wrongpattern"
            )
        with self.assertRaises(Exception):
            fname = feature_extraction.resizer(
                self.list_test, self.endpath.name, "Image2"
            )
        with self.assertRaises(Exception):
            self.list_test_wrong = ["wrongpath", "wrongpathmask"]
            fname = feature_extraction.resizer(
                self.list_test_wrong, self.endpath.name, "Image1"
            )


if __name__ == "__main__":
    unittest.main()
