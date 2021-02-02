from kerastuner.tuners import RandomSearch, BayesianOptimization
from kerastuner.engine.hyperparameters import HyperParameters

import time
LOG_DIR = f"{int(time.time())}"

from keras.constraints import unit_norm,min_max_norm,max_norm
from tensorflow.keras import regularizers


def build_model_rad_UNET(hp,shape=(124,124,1),feature_dim=(3,)):
    input_tensor = Input(shape=shape)
    input_vector= Input(shape=feature_dim)

    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(input_tensor)
    c1 = Dropout(0.2)(c1)
    c1 = Conv2D(16, (3, 3),activation='relu', kernel_initializer='he_normal',
                            padding='same')(c1)
    p1 =MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Resizing(32,32,interpolation='nearest')(p2)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(p2)


    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Resizing(16,16,interpolation='nearest')(p3)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(c4)

    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(p4)

    c5 = Dropout(0.2)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same', name="last_conv")(c5)
#fc layers

    flat=Flatten()(c5)
    flat=concatenate([flat,input_vector])
    den = Dense(hp.Int(f'dense_base_unit',
                                min_value=32,
                                max_value=64,
                                step=4), activation='relu', activity_regularizer=regularizers.l2(1e-4))(flat)

    for i in range(hp.Int('n_layers', 1, 4)):  # adding variation of layers.


      den = Dense(hp.Int(f'conv_{i}_units',
                                min_value=4,
                                max_value=64,
                                step=4), activation='relu', activity_regularizer=regularizers.l2(1e-4))(den)
      den= Dropout(hp.Float(f'drop_{i}_rate',
                                min_value=0,
                                max_value=.5,
                                step=.1,))(den)

    classification_output = Dense(2, activation = 'sigmoid', name="classification_output")(den)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)

    #c4 = Resizing(14,14,interpolation='nearest')(c4)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    #c3= Resizing(28,28,interpolation='nearest')(c3)

    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = Resizing(62,62,interpolation='nearest')(c2)

    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(u8)
    c8 = Dropout(0.2)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    #c1= Resizing(112,112,interpolation='nearest')(c1)

    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(u9)
    c9 = Dropout(0.2)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(c9)


    decoder_out = Conv2D(1, (1, 1), activation='sigmoid',name="decoder_output")(c9)

    model = Model([input_tensor,input_vector], [decoder_out,classification_output])
    model.compile(optimizer='adam', loss={'decoder_output':'binary_crossentropy','classification_output':'categorical_crossentropy'},
                  metrics={'decoder_output':'MAE','classification_output':tf.keras.metrics.AUC()})
    return model

tuner = BayesianOptimization(
    build_model_rad_UNET,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    directory=LOG_DIR)

tuner.search(mass_gen_rad,
             verbose=2,
             epochs=250,
             batch_size=len(mass_gen_rad),
             #callbacks=[tensorboard],
             validation_data=([X_train_rad_val,feature_train_val], [Y_train_rad_val,class_train_rad_val]))

tuner.get_best_hyperparameters()[0].values

newmod=build_modelUNET(tuner.get_best_hyperparameters()[0])
#newmod.summary()

tuner.get_best_models()[0].summary()

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

newmod.compile(optimizer='adam', loss={'decoder_output':'binary_crossentropy','classification_output':'binary_crossentropy'}, metrics={'decoder_output':'MAE','classification_output':tf.keras.metrics.AUC()})

history_tuned = newmod.fit(mass_gen, steps_per_epoch=len(mass_gen), epochs=200, validation_data=(X_train_val, [Y_train_val,class_train_val]),callbacks=[tensorboard_callback])

"""View on tensorboard"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir logs

"""Save the model"""

newmod.save('/content/drive/My Drive/model_optV3') #the model_opt is better

modelviewer(history_tuned)