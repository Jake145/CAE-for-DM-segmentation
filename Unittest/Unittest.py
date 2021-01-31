import unittest
import mock
import sys
import numpy as np
import unittest
import tempfile
import os
import mock
import pickle
import shutil
from PIL import Image
sys.path
sys.path.append('C:/Users/pensa/Desktop/CAE-for-DM-segmentation/functioncae')
import caehelper

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

file_handler = logging.FileHandler('Unittest.log')
file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


class Test_CAE(unittest.TestCase):

    #setup i metodi setup e teardown
    @classmethod
    def setUpClass(cls):

        #setup per il dataset piccolo
        cls.shape_im=(124,124)
        cls.image_ones=np.ones(cls.shape_im)
        cls.image_zeros=np.zeros(cls.shape_im)
        cls.image_square=np.zeros(cls.shape_im)
        cls.image_square[50:100,50:100]=1
        cls.image_mask=cls.image_square*255

        cls.temp_dir=tempfile.TemporaryDirectory()

        cls.filename1='Image1_benign_resized.png'
        cls.path_1=os.path.join(cls.temp_dir.name,cls.filename1 )
        cls.new_img_1 = Image.fromarray(cls.image_ones)
        cls.new_img_1 = cls.new_img_1.convert("L")
        cls.new_img_1.save(cls.path_1)

        cls.filename2='Image2_malign_resized.png'
        cls.path_2=os.path.join(cls.temp_dir.name, cls.filename2)
        cls.new_img_2 = Image.fromarray(cls.image_zeros)
        cls.new_img_2 = cls.new_img_2.convert("L")
        cls.new_img_2.save(cls.path_2)

        cls.filename3='Image2_malign_mass_mask.png'
        cls.path_3=os.path.join(cls.temp_dir.name, cls.filename3)
        cls.new_img_3 = Image.fromarray(cls.image_square)
        cls.new_img_3 = cls.new_img_3.convert("L")
        cls.new_img_3.save(cls.path_3)

        cls.filename4='Image1_benign_mass_mask.png'
        cls.path_4=os.path.join(cls.temp_dir.name, cls.filename4)
        cls.new_img_4 = Image.fromarray(cls.image_mask)
        cls.new_img_4 = cls.new_img_4.convert("L")
        cls.new_img_4.save(cls.path_4)

        cls.filename5='Only_Ones.pgm'
        cls.path_5=os.path.join(cls.temp_dir.name, cls.filename5)
        cls.new_img_5 = Image.fromarray(cls.image_ones)
        cls.new_img_5 = cls.new_img_5.convert("L")
        cls.new_img_5.save(cls.path_5)


        #setup per le immagini grandi

        cls.shape_im_big=(5000,3000)
        cls.image_ones_big=np.ones(cls.shape_im_big)
        cls.image_zeros_big=np.zeros(cls.shape_im_big)
        cls.image_square_big=np.zeros(cls.shape_im_big)
        cls.image_square_big[500:1000,500:1000]=1
        cls.image_mask_big=cls.image_square_big*255

        cls.temp_dir_big_mass=tempfile.TemporaryDirectory()

        cls.filename1_big='Image1_benign.png'
        cls.path_1_big=os.path.join(cls.temp_dir_big_mass.name,cls.filename1_big )
        cls.new_img_1_big = Image.fromarray(cls.image_ones_big)
        cls.new_img_1_big = cls.new_img_1_big.convert("L")
        cls.new_img_1_big.save(cls.path_1_big)

        cls.filename2_big='Image2_malign.png'
        cls.path_2_big=os.path.join(cls.temp_dir_big_mass.name, cls.filename2_big)
        cls.new_img_2_big = Image.fromarray(cls.image_zeros_big)
        cls.new_img_2_big = cls.new_img_2_big.convert("L")
        cls.new_img_2_big.save(cls.path_2_big)

        cls.temp_dir_big_masks=tempfile.TemporaryDirectory()

        cls.filename3_big='Image1_benign.png'
        cls.path_3_big=os.path.join(cls.temp_dir_big_masks.name, cls.filename3_big)
        cls.new_img_3_big = Image.fromarray(cls.image_square_big)
        cls.new_img_3_big = cls.new_img_3_big.convert("L")
        cls.new_img_3_big.save(cls.path_3_big)

        cls.filename4_big='Image2_malign.png'
        cls.path_4_big=os.path.join(cls.temp_dir_big_masks.name, cls.filename4_big)
        cls.new_img_4_big = Image.fromarray(cls.image_mask_big)
        cls.new_img_4_big = cls.new_img_4_big.convert("L")
        cls.new_img_4_big.save(cls.path_4_big)

        #cartella pickle
        cls.pickles=tempfile.TemporaryDirectory()
        cls.pickle={'thispickle':'thatpickle'}
        cls.pickle_filename='Pick.pickle'
        cls.pickle_path=os.path.join(cls.pickles.name,cls.pickle_filename)
        with open(cls.pickle_path,'wb') as handle:
            pickle.dump(cls.pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()
        cls.temp_dir_big_mass.cleanup()
        cls.temp_dir_big_masks.cleanup()
        cls.pickles.cleanup()

    def setUp(self):
        self.endpath=tempfile.TemporaryDirectory()

    def tearDown(self):
        self.endpath.cleanup()


    def test_save_newext(self):
        stat=caehelper.save_newext(self.filename5,self.temp_dir.name,
                                        'pgm','png',self.endpath.name)
        self.assertTrue(stat)

        with self.assertRaises(Exception):
            caehelper.save_newext('failfile.pgm',self.temp_dir.name,
                                        'pgm','png',self.endpath.name)


    def test_unit_mask(self):
        stat,image=caehelper.unit_masks(self.filename4,self.temp_dir.name,
                                        'png','pgm',self.endpath.name)
        self.assertTrue(stat)
        np.testing.assert_allclose(np.round(image),self.image_zeros) #qui entra in gioco dei problemi con arrotondamento
        with self.assertRaises(Exception):
            caehelper.unit_masks('failfile.pgm',self.temp_dir.name,
                                        'pgm','png',self.endpath.name)
    #@mock.patch('caehelper.os.listdir') using mock is good to use for the heatmap and modelviewer maybe
    def test_read_dataset(self):
        X,Y,CLASS=caehelper.read_dataset(self.temp_dir.name,'png',
        'benign','malign',x_id ="_resized", y_id="_mass_mask")
        self.assertEqual(len(X),len(Y))
        self.assertEqual(len(X),len(CLASS))
        self.assertEqual(list(CLASS),[0,1])
        with self.assertRaises(Exception):
            X,Y,CLASS=caehelper.read_dataset(self.temp_dir.name,'jpg',
                    'benign','malign',x_id ="_resized", y_id="_mass_mask")
        with self.assertRaises(Exception):
            X,Y,CLASS=caehelper.read_dataset('C:\Mock\fakedirpath','png',
                    'benign','malign',x_id ="_resized", y_id="_mass_mask")
        with self.assertRaises(Exception):
            X,Y,CLASS=caehelper.read_dataset(self.temp_dir.name,'png',
                    'benign','malign',x_id ="_ReSiz3D", y_id="_mess_mansk")

    def test_read_dataset_big(self):

        X,Y,CLASS=caehelper.read_dataset_big(self.temp_dir_big_mass.name,self.temp_dir_big_masks.name,'benign','malign',ext='png')
        self.assertEqual(len(X),len(Y))
        self.assertEqual(len(X),len(CLASS))
        self.assertEqual(list(CLASS),[0,1])
        with self.assertRaises(Exception):
            X,Y,CLASS=caehelper.read_dataset_big(self.temp_dir_big_mass.name,self.temp_dir_big_masks.name,'benign','malign',ext='jpg',resize=True)
        with self.assertRaises(Exception):
            X,Y,CLASS=caehelper.read_dataset_big('C:\Mock\fakedirpath','C:\Mock\fakedirpath2','benign','malign',ext='png',resize=True)


    def test_dict_update_radiomics(self):
        mockdict={}
        diz=caehelper.dict_update_radiomics(self.pickle_path,mockdict)
        self.assertEqual(len(diz),1)

    def test_blender(self):
        image=caehelper.blender(self.image_ones,self.image_ones,1,1)
        np.testing.assert_array_equal(image,np.full(self.shape_im, 2))
        with self.assertRaises(Exception):
            caehelper.blender('fake.png','string.txt','really wrong','oh noes')

    def test_dice(self):
        dix=caehelper.dice(self.image_ones,self.image_square)
        dix_1=caehelper.dice(self.image_ones,self.image_ones)
        dix_0=caehelper.dice(self.image_zeros,self.image_ones)
        self.assertEqual(np.round(dix,2),0.28)
        self.assertEqual(np.round(dix_1),1)
        self.assertEqual(np.round(dix_0),0)

    def test_modelviewer(self):
        with self.assertRaises(Exception):
            caehelper.modelviewer('nothing')

    def test_otsu(self):
        masked=caehelper.otsu(0.5*self.image_square)
        np.testing.assert_array_equal(masked,self.image_square)






if __name__ == '__main__':
    unittest.main()
