"""docstring"""
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

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

file_handler = logging.FileHandler("RadiomicsSegm.log")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


DATAPATH = (
    "C:/Users/pensa/Desktop/CAE-for-DM-segmentation/large_sample_Im_segmented_ref"
)

images_rad, masks_rad, labels_rad = caehelper.read_dataset(
    DATAPATH, "pgm", "_2_resized", "_1_resized"
)
images_rad = images_rad / 255
masks_rad = masks_rad / 255



IMAGES_ID = "_resized"
MASKS_ID = "_mass_mask"
EXT = "pgm"
fnames = glob.glob(os.path.join(DATAPATH, f"*{IMAGES_ID}.{EXT}"))
fnamesmask = glob.glob(os.path.join(DATAPATH, f"*{MASKS_ID}.{EXT}"))

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
    f.replace(DATAPATH, ""): extractor.execute(
        caehelper.read_pgm_as_sitk(f),
        caehelper.read_pgm_as_sitk(f.replace(IMAGES_ID, MASKS_ID)),
        label=255,
    )
    for f in fnames
}


Pandata = pd.DataFrame(dataframe)


# for i, name in enumerate(Pandata.index):
#    if "diagnostics" in Pandata.index[i]:
#        print(i)
#    else:
#        pass


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


exp_var_cumsum = pd.Series(np.round(pca.explained_variance_ratio_.cumsum(), 4) * 100)
for index, var in enumerate(exp_var_cumsum):
    print("if n_components= %d,   variance=%f" % (index, np.round(var, 3)))


pca = PCA(n_components=3)
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
# transform


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


##
model_rad = cae_cnn_models.make_model_rad_regulizer(
    batch[0][0].shape[1:], batch[0][1].shape[1:]
)
model_rad.summary()
MAINPATH = "C:/Users/pensa/Desktop/CAE-for-DM-segmentation/models/"
FILENAME = "rad_reg_weights.{epoch:02d}-{val_loss:.2f}.h5"
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
    metrics={"decoder_output": "MAE", "classification_output": tf.keras.metrics.AUC()},
)

EPOCH_NUMBER = 100

HISTORY_RAD = model_rad.fit(
    mass_gen_rad,
    steps_per_epoch=len(mass_gen_rad),
    epochs=EPOCH_NUMBER,
    validation_data=(
        [mass_train_rad_val, feature_train_val],
        [mask_train_rad_val, class_train_rad_val],
    ),
    callbacks=[model_checkpoint_callback],
)

caehelper.modelviewer(HISTORY_RAD)


# model_rad = keras.models.load_model(
#    "C:/Users/pensa/Desktop/CAE-for-DM-segmentation/models/rad_Unet_weights.17-0.56.h5"
# )


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
        model_rad.predict([xtrain, feature_train[IDX][np.newaxis, ...]])[0].squeeze()
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
    mask_test_rad, caehelper.otsu(model_rad.predict([mass_test_rad, feature_test])[0])
).mean()

hmap = caehelper.heatmap_rad(mass_test_rad[18], feature_test[18], model_rad)

y_pred = model_rad.predict([mass_test_rad, feature_test])[1]
fpr, tpr, thresholds = roc_curve(
    class_test_rad, [item[0] for _, item in enumerate(y_pred)], pos_label=0
)

auc = auc(fpr, tpr)

caehelper.plot_roc_curve(fpr, tpr, auc)
