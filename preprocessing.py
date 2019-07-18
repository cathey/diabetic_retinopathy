# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 15:12:06 2019

@author: Cathey
"""

import os
import random
import pandas as pd
import numpy as np
import cv2


"""
Divide data in directory into sub_dirs based on labels from csv file
"""
def divide_data(train_dir, val_dir, train_labels, N, classes):
#    C = classes.size
    os.mkdir(val_dir)
    for c in classes:
        os.mkdir(os.path.join(train_dir, str(c)))
        os.mkdir(os.path.join(val_dir, str(c)))
    
    train_class = []
    val_class = []
    for i in range(N):
        c = train_labels['diagnosis'][i]            # class
        fn = train_labels['id_code'][i]+'.png'      # file name
        if random.random() > 0.9:                   # move to val set
            os.rename(os.path.join(train_dir, fn), os.path.join(val_dir, str(c), fn))
            val_class.append(c)
        else:
            os.rename(os.path.join(train_dir, fn), os.path.join(train_dir, str(c), fn))
            train_class.append(c)

    for c in classes:
        N_train = len(os.listdir(os.path.join(train_dir, str(c))))
        N_val = len(os.listdir(os.path.join(val_dir, str(c))))
        print('Class ' + str(c) + ': N_train = ' + str(N_train) + ', N_val = ' + str(N_val))
        
    return np.array(train_class), np.array(val_class)


"""
Combine data in all subdirs of train & test dirs
"""
def combine_data(train_dir, val_dir, classes):
    for c in classes:
        all_train = os.listdir(os.path.join(train_dir, str(c)))
        all_val = os.listdir(os.path.join(val_dir, str(c)))
        for fn in all_train:
            os.rename(os.path.join(train_dir, str(c), fn), os.path.join(train_dir, fn))
        for fn in all_val:
            os.rename(os.path.join(val_dir, str(c), fn), os.path.join(train_dir, fn))
        os.rmdir(os.path.join(train_dir, str(c)))
        os.rmdir(os.path.join(val_dir, str(c)))
    os.rmdir(val_dir)


"""
remove the black borders of an img
"""
def cut_black(img, tol=5):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = img_gray > tol
    idx = np.ix_(mask.any(1),mask.any(0))
    return img[idx[0], idx[1], :]


"""
Only take the center
"""
def crop_center(img):
    H, W = img.shape[0], img.shape[1]
    if H == W:
        return img
    elif H > W:
        return img[H//2-W//2:H//2+W//2, :, :]
    else:
        return img[:, W//2-H//2:W//2+H//2, :]
        
        
"""
Adjust brightness, scale to mean 100
"""
def adjust_light(img):
    brightness = np.mean(img)
    return np.clip(100.0/brightness*img, 0, 255).astype(int)


if __name__ == "__main__":
    train_dir = "..\\data\\train"
    test_dir = "..\\data\\test"
    train_dir_processed = "..\\data\\train_processed"
    test_dir_processed = "..\\data\\test_processed"
    train_label_file = "..\\train.csv"
    test_name_file = "..\\test.csv"
    test_pred_file = "..\\test_pred.csv"
    
    train_labels = pd.read_csv(train_label_file)
    classes = train_labels['diagnosis'].unique()
    
    global N, C, L
    N = train_labels.shape[0]
    C = classes.size
    L = 224
    
#    combine_data(train_dir, val_dir, C)
    
    print("process train")
    train_imgs = os.listdir(train_dir)
    for train_img in train_imgs:
        img = cv2.imread(os.path.join(train_dir, train_img))
        img = cut_black(img)
        img = crop_center(img)
        img = adjust_light(img)
        cv2.imwrite(os.path.join(train_dir_processed, train_img), img)
        
    print("process test")
    test_imgs = os.listdir(test_dir)
    for test_img in test_imgs:
        img = cv2.imread(os.path.join(test_dir, test_img))
        img = cut_black(img)
        img = crop_center(img)
        img = adjust_light(img)
        cv2.imwrite(os.path.join(test_dir_processed, test_img), img)