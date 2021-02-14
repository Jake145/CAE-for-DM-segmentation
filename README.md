# CAE-for-DM-segmentation
Convolutional autoencoder for the CMEP course. This project consists of three versions of convolutional autoencoders with classification of the masses. The first classifies the masses in /large_sample_Im_segmented_ref based only on pixel values, while the second also takes in features obtained with pyradiomics into account.
The third one is and attempt to adapt the net to ta very large dataset from TCIA (https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM) using multithreading, multiprocessing and special classes to flow the data into the CAE.
For all the nets there is a colab notebook and specific python files.
# Special package
Included is a specific package with the models, classes and helper functions. These range from simple data processing and I/O operations to class activation map visualization.
#Dataset
The smaller dataset is included, while the TCIA dataset can be downloaded from the link above and preprocessed first with dycomdatagen.py to create .png datasets and feature_extraction.py to extract the radiomic features, which are in Pandabigframe.csv. For ease a shared google drive will be included to run the notebook version.
In these scripts multitreading and multiprocessing is used to accelerate the operations.
# Models
There are three proposed models, a simple one, one with added regularization and finally a unet. Models can be further optimized by modifing the TUNER.ipynb notebook, in which Bayesian optimization is used.
[![Build Status](https://www.travis-ci.com/Jake145/CAE-for-DM-segmentation.svg?branch=main)](https://www.travis-ci.com/Jake145/CAE-for-DM-segmentation)
[![Documentation Status](https://readthedocs.org/projects/cae/badge/?version=latest)](https://cae.readthedocs.io/en/latest/?badge=latest)
https://img.shields.io/pypi/v/CAE-Jake-HP-145