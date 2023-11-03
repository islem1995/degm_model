# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 22:09:43 2023

@author: laptop zone
"""

import numpy as np
import tensorflow as tf
from sklearn.mixture import GaussianMixture
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.models import Model
from utils import get_ACC_NMI
from utils import get_xy
from utils import log_csv
import time
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info and warning messages


def model_conv(load_weights=True):
    init = 'uniform'
    filters = [32, 64, 128, hidden_units]
    if input_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'
    input = layers.Input(shape=input_shape)
    x = layers.Conv2D(filters[0], kernel_size=5, strides=2, padding='same', activation='relu', kernel_initializer=init)(input)
    x = layers.Conv2D(filters[1], kernel_size=5, strides=2, padding='same', activation='relu', kernel_initializer=init)(x)
    x = layers.Conv2D(filters[2], kernel_size=3, strides=2, padding=pad3, activation='relu', kernel_initializer=init)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units=filters[-1], name='embed')(x)
    h = x
    x = layers.Dense(filters[2] * (input_shape[0] // 8) * (input_shape[0] // 8), activation='relu')(x)
    x = layers.Reshape((input_shape[0] // 8, input_shape[0] // 8, filters[2]))(x)
    x = layers.Conv2DTranspose(filters[1], kernel_size=3, strides=2, padding=pad3, activation='relu')(x)
    x = layers.Conv2DTranspose(filters[0], kernel_size=5, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(input_shape[2], kernel_size=5, strides=2, padding='same')(x)
    output = layers.Concatenate()([h, layers.Flatten()(x)])
    model = Model(inputs=input, outputs=output)
    # model.summary()
    if load_weights:
        model.load_weights(f'weight_base_{ds_name}.h5')
        print('model_conv: weights were loaded')
    return model

def loss_train_base(y_true, y_pred):
    y_true = layers.Flatten()(y_true)
    y_pred = y_pred[:, hidden_units:]
    return losses.mse(y_true, y_pred)

def train_base(ds_xx):
    model = model_conv(load_weights=False)
    model.compile(optimizer='adam', loss=loss_train_base)
    model.fit(ds_xx, epochs=pretrain_epochs, verbose=2)
    model.save_weights(f'weight_base_{ds_name}.h5')

def sorted_eig(X):
    e_vals, e_vecs = np.linalg.eig(X)
    idx = np.argsort(e_vals)
    e_vecs = e_vecs[:, idx]
    e_vals = e_vals[idx]
    return e_vals, e_vecs
#-----------------------------------------------------------------------------------------------
def train(x, y):
    log_str = f'iter; acc, nmi, ri ; loss; n_changed_assignment; time:{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}'
    log_csv(log_str.split(';'), file_name=ds_name)
    #log_str = f'iter; acc, nmi, loss; n_changed_assignment; time:{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}'

    model = model_conv()

    optimizer = tf.keras.optimizers.Adam()
    loss_value = 0
    index = 0

    assignment = np.array([-1] * len(x))
    index_array = np.arange(x.shape[0])

    n_change_assignment = 0  # Initialize n_change_assignment here
    H_vt = None  # Initialize H_vt here
    U_vt = None  # Initialize U_vt here

    #for ite in range(10):
    for ite in range(int(1 * 10)):
        if ite % update_interval == 0:
            H = model(x).numpy()[:, :hidden_units]
    #for ite in range(10):
        #if ite % update_interval == 0:
           # H = model(x).numpy()[:, :hidden_units]
    

        #if ite >= 10 * update_interval:  
            #break
        #if ite == 0:

            # Use Gaussian Mixture Model (GMM) for clustering
            gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=0)
            gmm.fit(H)
            assignment_new = gmm.predict(H)

            n_change_assignment = np.sum(assignment_new != assignment)
            assignment = assignment_new

            if n_change_assignment <= len(x) * 0.005:
              model.save_weights(f'weight_final_gmm_{ds_name}.h5')
              print('end')
              break


            S_i = []
            for i in range(n_clusters):
                temp = H[assignment == i] - gmm.means_[i]
                temp = np.matmul(np.transpose(temp), temp)
                S_i.append(temp)
            S_i = np.array(S_i)
            S = np.sum(S_i, 0)
            Evals, V = sorted_eig(S)
            H_vt = np.matmul(H, V)
            U_vt = np.matmul(gmm.means_, V)

            loss = np.round(np.mean(loss_value), 5)
            acc, nmi = get_ACC_NMI(np.array(y), np.array(assignment))

            # Log metrics
            log_str = f'iter {ite // update_interval}; acc, nmi,loss = {acc, nmi ,loss}; loss:' \
                      f'{loss:.5f}; n_changed_assignment:{n_change_assignment}; time:{time.time() - time_start:.3f}'
            #log_str = f'iter {ite // update_interval}; acc, nmi, loss = {acc, nmi, loss:.5f}; ' \
            print(log_str)
            log_csv(log_str.split(';'), file_name=ds_name)

            #if n_change_assignment <= len(x) * 0.005:
             #model.save_weights(f'weight_final_gmm_{ds_name}.h5')
             #print('end')
             #break

        idx = index_array[index * batch_size: min((index + 1) * batch_size, x.shape[0])]
        y_true = H_vt[idx]
        temp = assignment[idx]
        for i in range(len(idx)):
            y_true[i, -1] = U_vt[temp[i], -1]

        with tf.GradientTape() as tape:
            tape.watch(model.trainable_variables)
            y_pred = model(x[idx])
            y_pred_cluster = tf.matmul(y_pred[:, :hidden_units], V)
            loss_value = losses.mse(y_true, y_pred_cluster)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0


#def train(x, y):
  #  log_str = f'iter; acc, nmi, ri ; loss; n_changed_assignment; time:{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}'
    #log_csv(log_str.split(';'), file_name=ds_name)
    #model = model_conv()

    #optimizer = tf.keras.optimizers.Adam()

# GMM Training
#def train_gmm(features, n_components):
   # gmm = GaussianMixture(n_components=n_components, random_state=0)
   # gmm.fit(features)
   # return gmm

if __name__ == '__main__':
    pretrain_epochs = 2
    pretrain_batch_size = 256
    batch_size = 256
    update_interval = 10
    hidden_units = 10

    parser = argparse.ArgumentParser(description='select dataset: MNIST, COIL20, USPS')
    parser.add_argument('ds_name', default='USPS')
    args = parser.parse_args()
    if args.ds_name is None or not args.ds_name in ['MNIST','USPS','COIL20']:
        ds_name = 'USPS'
    else:
        ds_name = args.ds_name

    if ds_name == 'MNIST':
        input_shape = (28, 28, 1)
        n_clusters = 10
    elif ds_name == 'USPS':
        input_shape = (16, 16, 1)
        n_clusters = 10
    elif ds_name == 'COIL20':
        input_shape = (28, 28, 1)
        n_clusters = 20

    time_start = time.time()
    x, y = get_xy(ds_name=ds_name)
    ds_xx = tf.data.Dataset.from_tensor_slices((x, x)).shuffle(8000).batch(pretrain_batch_size)
    train_base(ds_xx)
    train(x, y)
    print(time.time() - time_start)