# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 22:10:58 2023

@author: laptop zone
"""

import numpy as np
import tensorflow as tf
from sklearn.mixture import GaussianMixture
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.models import Model
from sklearn.decomposition import PCA
from utils import get_ACC_NMI
from utils import get_xy
from utils import log_csv
import time
import argparse

#def model_conv(input_shape, hidden_units, load_weights=True):
def model_conv(load_weights=True):
    filters = [500, 500, 2000]
    init = 'uniform'
    activation = 'relu'
    input = layers.Input(shape=input_shape)
    x = input
    for i in range(len(filters)):
        x = layers.Dense(filters[i], activation=activation, kernel_initializer=init)(x)
    x = layers.Dense(hidden_units, kernel_initializer=init)(x)
    # x = tf.divide(x, tf.expand_dims(tf.norm(x, 2, -1), -1))
    h = x

    for i in range(len(filters) - 1, -1, -1):
        x = layers.Dense(filters[i], activation=activation, kernel_initializer=init)(x)
    y = layers.Dense(input_shape, kernel_initializer=init)(x)

    output = layers.Concatenate()([h, y])
    model = Model(inputs= input, outputs=output)
    if load_weights:
        model.load_weights(f'weight_base_{ds_name}.h5')
        print('model_conv: weights were loaded')
    return model

def loss_train_base(y_true, y_pred):
    y_true = layers.Flatten()(y_true)
    y_pred = y_pred[:, hidden_units:]
    return losses.mse(y_true, y_pred)

def train_base(ds_xx):
    def reshape_and_cast(x, y):
     # Convert x to a NumPy array and cast it to float32
     ds_xx = np.array(x, dtype=np.float32)
     # Reshape x
     ds_xx = tf.reshape(x, (x.shape[0], input_shape))
     return x, y
    # Apply reshape and casting to the dataset
     x = ds_xx.map(reshape_and_cast)
     x = ds_xx.shuffle(8000).batch(pretrain_batch_size)
    #model = model_conv(input_shape,hidden_units, load_weights=False)
    model = model_conv(load_weights=False)
    model.compile(optimizer='adam',loss=loss_train_base)
    model.fit(ds_xx, epochs=pretrain_epochs, verbose=2)
    model.save_weights(f'weight_base_{ds_name}.h5')

def sorted_eig(X):
    e_vals, e_vecs = np.linalg.eig(X)
    idx = np.argsort(e_vals)
    e_vecs = e_vecs[:, idx]
    e_vals = e_vals[idx]
    return e_vals, e_vecs

#def train(x, y, model):
def train(x, y):
  #if __name__ == '__main__':
    #pretrain_epochs = 2
    #pretrain_batch_size = 256
    #batch_size = 256
    #update_interval = 10
  # Set the input_shape and hidden_units based on your dataset
    #if ds_name == 'MNIST':
        #input_shape = 784
        #hidden_units = 50
    #elif ds_name == 'USPS':
        #input_shape = 256
        #hidden_units = 50
    #elif ds_name == 'COIL20':
        #input_shape = 28 * 28
        #hidden_units = 50

    log_str = f'iter; acc, nmi, ri ; loss; n_changed_assignment; time:{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}'
    log_csv(log_str.split(';'), file_name=ds_name)

    # Call model_conv with input_shape and hidden_units
    model = model_conv(input_shape, hidden_units)

    optimizer = tf.keras.optimizers.Adam()
    loss_value = 0
    index = 0

    assignment = np.array([-1] * len(x))
    index_array = np.arange(x.shape[0])

    #for ite in range(int(140 * 100)):
        #if ite % update_interval == 0:
            #H = model(x).numpy()[:, :hidden_units]
    for ite in range(int(140 * 100)):
         if ite % update_interval == 0:
            H = model(x).numpy()[:, :hidden_units]


        # Rest of your code for clustering, metrics, and logging

            # Use Gaussian Mixture Model (GMM) for clustering
            gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=0)
            gmm.fit(H)
            assignment_new = gmm.predict(H)

            n_change_assignment = np.sum(assignment_new != assignment)
            assignment = assignment_new

    #optimizer = tf.keras.optimizers.Adam()
    #loss_value = 0
    #index = 0
    #-------------------------
    #gmm_n_components = 100

    #assignment = np.array([-1] * len(x))
    #index_array = np.arange(x.shape[0])

    # Reshape the input data using NumPy
    #x = x.reshape(x.shape[0], -1)
    #-------------------------

    #pca = PCA(n_components=hidden_units)

    #for ite in range(int(140 * 100)):
        #if ite % update_interval == 0:
            #H = model(x).numpy()[:, :hidden_units]
            #-------------------------
            #gmm = GaussianMixture(n_components=gmm_n_components, covariance_type='full', random_state=0).fit(H)
            #gmm.fit(H)
            #assignment_new = gmm.predict(H)

            #n_change_assignment = np.sum(assignment_new != assignment)
            #assignment = assignment_new

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
            log_str = f'iter {ite // update_interval}; acc, nmi, loss = {acc, nmi, loss:.5f}; ' \
                      f'n_changed_assignment:{n_change_assignment}; time:{time.time() - time_start:.3f}'
            print(log_str)
            log_csv(log_str.split(';'), file_name=ds_name)

     if n_change_assignment <= len(x) * 0.005:
            model.save_weights(f'weight_final_gmm_{ds_name}.h5')
            print('end')
        break

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

            #S_i = []
            #for i in range(gmm_n_components):
                #temp = H[assignment == i] - gmm.means_[i]
                #temp = np.matmul(np.transpose(temp), temp)
                #S_i.append(temp)
            #S_i = np.array(S_i)
            #S = np.sum(S_i, 0)
            #Evals, V = sorted_eig(S)
            #H_vt = np.matmul(H, V)

            #loss = np.round(np.mean(loss_value), 5)
            #acc, nmi = get_ACC_NMI(np.array(y), np.array(assignment))

            #log_str = f'iter {ite // update_interval}; acc, nmi, ri = {acc, nmi}; loss:' \
                      #f'{loss}; n_changed_assignment:{n_change_assignment}; time:{time.time() - time_start:.3f}'
            #print(log_str)
            #log_csv(log_str.split(';'), file_name=ds_name)

        #if n_change_assignment <= len(x) * 0.005:
            #model.save_weights(f'weight_final_{ds_name}.h5')
            #print('end')
            #break

        #idx = index_array[index * batch_size: min((index + 1) * batch_size, x.shape[0])]
        #y_true = H_vt[idx]
        #temp = assignment[idx]
        #for i in range(len(idx)):
         #   y_true[i, -1] = U_vt[temp[i], -1]

        #with tf.GradientTape() as tape:
            #tape.watch(model.trainable_variables)
            #y_pred = model(x[idx])
            #y_pred_cluster = tf.matmul(y_pred[:, :hidden_units], V)
            #loss_value = losses.mse(y_true, y_pred_cluster)
        #grads = tape.gradient(loss_value, model.trainable_variables)
        #optimizer.apply_gradients(zip(grads, model.trainable_variables))

        #index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0
#if __name__ == '__main__':
    #pretrain_epochs = 5
    #pretrain_batch_size = 256
    #batch_size = 256
    #update_interval = 10

    #parser = argparse.ArgumentParser(description='select dataset: MNIST, USPS, COIL20')
    #parser.add_argument('ds_name', default='MNIST')
    #args = parser.parse_args()
    #ds_name = args.ds_name
    #if args.ds_name is None or not args.ds_name in ['MNIST', 'USPS', 'COIL20']:

    #ds_name = 'MNIST'

    #else:
       #ds_name = args.ds_name

    #if ds_name == 'MNIST':
        #input_shape = 784
        #hidden_units = 50
    #if ds_name == 'USPS':
        #input_shape = 256
        #hidden_units = 50
    #elif ds_name == 'COIL20':
       # input_shape = 28 * 28
       # hidden_units = 50

    time_start = time.time()
    x, y = get_xy(ds_name=ds_name)

    x = x.reshape(x.shape[0], 784)

    ds_xx = tf.data.Dataset.from_tensor_slices((x, x)).shuffle(8000).batch(pretrain_batch_size)

    train_base(ds_xx)
    train(x, y)
    print(time.time() - time_start)
