# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 22:09:01 2023

@author: laptop zone
"""
import tensorflow as tf
import numpy as np
import csv
import os
from tensorflow.keras.datasets import mnist
import h5py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info and warning messages

def get_ACC_NMI(_y, _y_pred):
    y = np.array(_y)
    y_pred = np.array(_y_pred)
    s = np.unique(y_pred)
    t = np.unique(y)

    N = len(np.unique(y_pred))
    C = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        for j in range(N):
            idx = np.logical_and(y_pred == s[i], y == t[j])
            C[i][j] = np.count_nonzero(idx)
    Cmax = np.amax(C)
    C = Cmax - C
    from scipy.optimize import linear_sum_assignment
    row, col = linear_sum_assignment(C)
    count = 0
    for i in range(N):
        idx = np.logical_and(y_pred == s[row[i]], y == t[col[i]])
        count += np.count_nonzero(idx)
    acc = np.round(1.0 * count / len(y), 5)

    temp = np.array(y_pred)
    for i in range(N):
        y_pred[temp == col[i]] = i
    from sklearn.metrics import normalized_mutual_info_score
    nmi = np.round(normalized_mutual_info_score(y, y_pred), 5)
    return acc, nmi

def get_xy(ds_name='MNIST', dir_path='datasets/', log_print=True, shuffle_seed=None):
    dir_path = os.path.join(dir_path, ds_name)  # Use os.path.join for directory paths

    if ds_name == 'MNIST':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, y_test))
        x = np.expand_dims(np.divide(x, 255.), -1)
    elif ds_name == 'USPS':
        try:
            with h5py.File(os.path.join(dir_path, 'USPS.h5'), 'r') as hf:
                x = hf['data'][:]
                y = hf['target'][:]
                x = x.astype(np.float32)  # Convert data to float32
                x = x / 255.0  # Normalize the data
        except Exception as e:
            print(f"Error loading USPS dataset: {e}")
            x, y = None, None  # Handle loading error gracefully

            # Convert data to float32
        x = x.astype(np.float32)

        # Normalize the data (you may need to adjust the normalization method)
        x = x / 255.0
        
    # ... (other dataset loading code for 'COIL20') ...
    elif ds_name == 'COIL20':
        # Load the COIL20 dataset using h5py
        f = h5py.File(os.path.join(dir_path, 'COIL20.h5'), 'r')
        x = np.array(f['data'][()]).squeeze()
        x = np.expand_dims(np.swapaxes(x, 1, 2).astype(np.float32), -1)
        x = tf.image.resize(x, [28, 28]).numpy()
        x = x / 255.
        y = np.array(f['labels'][()]).astype(np.float32)
        y[y == 20.] = 0.

    if x is None or y is None:
        return None, None  # Handle the error case

    # Shuffle and print dataset (if required)
    if not shuffle_seed:
        shuffle_seed = int(np.random.randint(100))
    idx = np.arange(0, len(x))
    idx = tf.random.shuffle(idx, seed=shuffle_seed).numpy()
    x = x[idx]
    y = y[idx]

    if log_print:
        print(ds_name)

    return x, y


    if not shuffle_seed:
        shuffle_seed = int(np.random.randint(100))
    idx = np.arange(0, len(x))
    idx = tf.random.shuffle(idx, seed=shuffle_seed).numpy()
    x = x[idx]
    y = y[idx]
    # x = tf.random.shuffle(x, seed=shuffle_seed).numpy()
    # y = tf.random.shuffle(y, seed=shuffle_seed).numpy()

    if log_print:
        print(ds_name)

    return x, y



def log_csv(strToWrite, file_name):
    path = r'log_history/'
    if not os.path.exists(path):
        os.makedirs(path)
    f = open(path + file_name + '.csv', 'a+', encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(strToWrite)
    f.close()


def read_list(file_name, type='int'):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    if type == 'str':
        array = np.asarray([l.strip() for l in lines])
        return array
    elif type == 'int':
        array = np.asarray([int(l.strip()) for l in lines])
        return array
    else:
        print("Unknown type")
        return None