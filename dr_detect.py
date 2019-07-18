# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 23:22:30 2019

@author: Cathey
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pandas as pd
import numpy as np
import math
import cv2
import time

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt

from preprocessing import divide_data, combine_data


"""
Single conv layer in CNN model
"""
def conv_layer(x_in, filters, kernel_dim, drop_rate=0.0, batch_norm=True, max_pool=True):
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=(kernel_dim,kernel_dim), strides=(1,1), padding='same',
               kernel_initializer='he_normal')(x_in)
    x = tf.keras.layers.Activation('relu')(x)
    if drop_rate > 0.0:
        x = tf.keras.layers.Dropout(drop_rate)(x)
    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    if max_pool:
        x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
    return x
    

"""
Single dense/FC layer in CNN model
"""
def dense_layer(x_in, units, activation='tanh', drop_rate=0.0):
    x = tf.keras.layers.Dense(units, activation=activation)(x_in)
    if drop_rate > 0.0:
        x = tf.keras.layers.Dropout(rate = drop_rate)(x)
    return x


"""
CNN model, from scratch
"""
def cnn_model():
    # input
    x_in = tf.keras.layers.Input(shape = (L,L,3), name = 'input')
    
    # conv layers
    x = conv_layer(x_in, filters=32, kernel_dim=3, drop_rate=0.0, batch_norm=True, max_pool=True)
    x = conv_layer(x, filters=48, kernel_dim=3, drop_rate=0.0, batch_norm=True, max_pool=True)
    x = conv_layer(x, filters=64, kernel_dim=3, drop_rate=0.0, batch_norm=True, max_pool=True)
    x = conv_layer(x, filters=96, kernel_dim=3, drop_rate=0.0, batch_norm=True, max_pool=True)
    #x = conv_layer(x, filters=128, kernel_dim=3, drop_rate=0.0, batch_norm=True, max_pool=True)
    
    # dense layers
    x = tf.keras.layers.Flatten()(x)
    x = dense_layer(x, units=1024, activation='tanh', drop_rate=0.0)
    x = dense_layer(x, units=64, activation='tanh', drop_rate=0.0)
    
    # class pred
    x_out = tf.keras.layers.Dense(5, activation='softmax')(x)
    
    # compile model
    model = tf.keras.models.Model(inputs=x_in, outputs=x_out)
#    sgd = tf.keras.optimizers.SGD(learning_rate=0.001)
    adam = tf.keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=1e-6)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])
    
    return model


"""
Custom loss function
"""
def custom_loss(y_true, y_pred):
    class_true = K.cast(K.expand_dims(K.argmax(y_true, axis=-1), axis=-1), 'float32')   # y_true is one-hot
    i = np.array([[0, 1, 2, 3, 4]]).astype('float32')
    alpha = K.square(i - class_true) / 16.0
    # cross entropy
    loss1 = -K.sum(y_true * K.log(y_pred), axis=-1)
    # additional term to penalize worse predictions
    loss2 = -K.sum(alpha * (1-y_true) * K.log(1-y_pred), axis=-1)
    
    return loss1+loss2


"""
Custom eval metric: quadratic weighted kappa
"""
def qwk(y_true, y_pred):
    # compute confution matrix
    y_true_label = K.argmax(y_true, axis=-1)    # one-hot to class number
    y_pred_label = K.argmax(y_pred, axis=-1)
    confusion = tf.math.confusion_matrix(y_true_label, y_pred_label, num_classes=5, dtype='float32')
    
    # compute quadratic weight
    alpha = np.square([[i-j for i in range(5)] for j in range(5)]).astype('float32')

    # compute observed and expected matrix
    observed = confusion/tf.reduce_sum(confusion)  # count -> distribution
    P_pred = tf.reduce_sum(confusion, axis=0)/tf.reduce_sum(confusion)
    P_true = tf.reduce_sum(confusion, axis=1)/tf.reduce_sum(confusion)
    expected = tf.tensordot(P_true, P_pred, axes=0)
    
    # compute kappa
    kappa = 1 - tf.reduce_sum(tf.multiply(alpha, observed))/tf.reduce_sum(tf.multiply(alpha, expected))
    return kappa


"""
Use transfer learning
"""
def transfer_model():
    base_model = tf.keras.applications.VGG16(input_shape=(L,L,3), weights='imagenet', include_top=False)
#    base_model = tf.keras.applications.InceptionResNetV2(input_shape=(L,L,3), weights='imagenet', include_top=False)
#    base_model = tf.keras.applications.ResNet50(input_shape=(L,L,3), weights='imagenet', include_top=False)
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    
    # dense layers
#    x = tf.keras.layers.Flatten()(x)
#    x = tf.keras.layers.GlobalAveragePooling2D()(x)    # faster processing, omit spacial, vessel leakage can happen anywhere
    x = tf.keras.layers.GlobalMaxPooling2D()(x)         # you only need to find certain features once, other areas can be blank
#    x = dense_layer(x, units=512, activation='elu', drop_rate=0.25)
    x = dense_layer(x, units=64, activation='tanh', drop_rate=0.25)
    
    # class pred
    x_out = tf.keras.layers.Dense(5, activation='softmax')(x)
    
    # compile model
    model = tf.keras.models.Model(inputs=base_model.input, outputs=x_out)
#    sgd = tf.keras.optimizers.SGD(learning_rate=0.001)
    adam = tf.keras.optimizers.Adam(lr=5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    
#    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])
    model.compile(optimizer=adam, loss=custom_loss, metrics=['acc', qwk])
    
    return model
    
"""
Generate (and augment) data for model
"""
def gen_data(train_dir, val_dir, batch_size):
    print("Train set:")
    train_datagen = ImageDataGenerator(rescale=1.0/255.0)
#                                       rotation_range=360,  # an eyeball at any angle is still the same
#                                       horizontal_flip=True,        # left/right eye symmetry
#                                       vertical_flip=True)          # left/right eye symmetry
                                       #brightness_range=(0.5, 1))   # there are dark imgs
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        batch_size=batch_size,
                                                        class_mode='categorical',
                                                        target_size=(L, L))
    print("Val set:")
    val_datagen = ImageDataGenerator(rescale=1.0/255.0)
#                                       rotation_range=360,  # an eyeball at any angle is still the same
#                                       horizontal_flip=True,        # left/right eye symmetry
#                                       vertical_flip=True)          # left/right eye symmetry
                                       #brightness_range=(0.5, 1))   # there are dark imgs
    val_generator = val_datagen.flow_from_directory(val_dir,
                                                      batch_size=batch_size,
                                                      class_mode='categorical',
                                                      target_size=(L, L))
    
    return train_generator, val_generator


"""
Display training history
"""
def show_history(history):
    acc=history.history['acc']
    val_acc=history.history['val_acc']
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    
    epochs = list(range(1, len(acc)+1))
    plt.figure(1)
    plt.plot(epochs, loss, 'b', epochs, val_loss, 'r')
    plt.title('loss')
    plt.legend(('train', 'val'))
    
    plt.figure(2)
    plt.plot(epochs, acc, 'b', epochs, val_acc, 'r')
    plt.title('accuracy')
    plt.legend(('train', 'val'))


"""
Write test set output
"""
def output_test(test_dir):
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    test_gen = test_datagen.flow_from_directory(test_dir, target_size=(L, L))
    test_pred = model.predict_generator(test_gen, workers=1)    # workers!=1 will mess up the order
    test_pred_class = np.argmax(test_pred, axis=1)
    test_files = test_gen.filenames
    
    test_df = pd.DataFrame(data={'id_code': test_files, 'diagnosis': test_pred_class})
    test_df.to_csv('..\\test_pred.csv', index=False)
    
    
"""
Driver
"""
if __name__ == "__main__":
    train_dir = "..\\data\\train_processed"
    val_dir = "..\\data\\val_processed"
    test_dir = "..\\data\\test_processed"
    train_label_file = "..\\train.csv"
    test_name_file = "..\\test.csv"
    test_pred_file = "..\\test_pred.csv"
    
    train_labels = pd.read_csv(train_label_file)
    classes = train_labels['diagnosis'].unique()
    classes = np.sort(classes)
    
    global N, C, L
    N = train_labels.shape[0]
    C = classes.size
    L = 224
    
#    combine_data(train_dir, val_dir, classes)
#    train_class, val_class = divide_data(train_dir, val_dir, train_labels, N, classes)
    
    train_gen, val_gen = gen_data(train_dir, val_dir, batch_size=64)
    
    model = transfer_model()
    model.summary()
#    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath = '..\\weights-best.hdf5', 
                                                    monitor='val_loss', save_best_only=True)]
    
    history = model.fit_generator(train_gen,
                                  validation_data = val_gen,
                                  steps_per_epoch=math.ceil(train_gen.samples/train_gen.batch_size),
                                  epochs = 50,
                                  validation_steps=math.ceil(val_gen.samples/val_gen.batch_size),
                                  callbacks = callbacks,
                                  #use_multiprocessing=False,
                                  verbose = 1)
    
    show_history(history)
    
    val_pred = model.predict_generator(val_gen, workers=1)    # workers!=1 will mess up the order
    val_pred_class = np.argmax(val_pred, axis=1)
    plt.figure(3)
    plt.hist(val_pred_class, bins=5)
    
    output_test()
    