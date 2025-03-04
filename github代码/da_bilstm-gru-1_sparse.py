# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 17:41:10 2019
@author: Suraj
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import tensorflow as tf
import keras.backend as K

from keras import Input, Model
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras.layers import Dense, GRU, Bidirectional, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

# ---------------------- Random Seed Setup -----------------------
np.random.seed(1)
tf.set_random_seed(2)

# ---------------------- Custom Huber Loss -----------------------
def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    is_small_error = K.abs(error) <= delta
    squared_loss = 0.5 * K.square(error)
    linear_loss = delta * (K.abs(error) - 0.5 * delta)
    return K.mean(tf.where(is_small_error, squared_loss, linear_loss))

# ---------------------- Lorenz-96 and RK4 -----------------------
def rhs(ne, u, forcing):
    """
    Lorenz-96 right-hand side.
    ne: number of variables in Lorenz-96 system.
    u:  state vector (1D).
    forcing: forcing term.
    """
    v = np.zeros(ne + 3)
    v[2:ne + 2] = u
    v[1] = v[ne + 1]
    v[0] = v[ne]
    v[ne + 2] = v[2]

    r = v[1:ne + 1] * (v[3:ne + 3] - v[0:ne]) - v[2:ne + 2] + forcing
    return r

def rk4(ne, dt, u, forcing):
    """
    Fourth-order Runge-Kutta integration.
    ne: number of variables.
    dt: time step.
    u:  current state (1D array).
    forcing: forcing term (F).
    """
    r1 = rhs(ne, u, forcing)
    k1 = dt * r1

    r2 = rhs(ne, u + 0.5 * k1, forcing)
    k2 = dt * r2

    r3 = rhs(ne, u + 0.5 * k2, forcing)
    k3 = dt * r3

    r4 = rhs(ne, u + k3, forcing)
    k4 = dt * r4

    return u + (k1 + 2.0 * (k2 + k3) + k4) / 6.0

# ---------------------- Data Preparation ------------------------
def create_training_data(features, labels, m, n, lookback):
    """
    Convert (m x n) features into (m-lookback+1, lookback, n) shape,
    aligning labels accordingly.
    """
    y_train = [labels[i, :] for i in range(m)]
    y_train = np.array(y_train)

    x_train = np.zeros((m - lookback + 1, lookback, n))
    for i in range(m - lookback + 1):
        chunk = features[i, :]
        for j in range(1, lookback):
            chunk = np.vstack((chunk, features[i + j, :]))
        x_train[i, :, :] = chunk
    return x_train, y_train

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# ---------------------- Main Program ---------------------------
# 1) Basic parameters
ne = 40           # Dimension of Lorenz-96
npe = 400         # Number of ensemble samples
forcing = 10.0    # Lorenz-96 forcing
dt = 0.005
tmax = 10.0
nt = int(tmax / dt)

lookback = 1      # Time steps to look back
nf = 10           # Observation frequency in time
nb = nt // nf     # Number of observation times
time_obs_indices = [nf * k for k in range(nb + 1)]
tobs = np.linspace(0, tmax, nb + 1)

# Load data (adjust file name if needed)
data = np.load('lstm_data_sparse.npz')
utrue = data['utrue']  # shape: (ne, nt+1)
uobs = data['uobs']    # shape: (ne, nb+1)
uwe  = data['uwe']     # shape: (ne, npe, nt+1)

# 2) Choose how many observed variables
me = 4
freq = ne // me
oin = [freq * i - 1 for i in range(1, me + 1)]

# da_model = 3 => use (ne + me) features
nfeat = ne + me

# 3) Build training set
for n in range(npe):
    # Features: (uwe at observation times) + (uobs for observed indices)
    fmat = np.hstack((uwe[:, n, time_obs_indices].T, uobs[oin, :].T))  # shape: (nb+1, nfeat)
    # Labels:   true - prior
    lmat = utrue[:, time_obs_indices].T - uwe[:, n, time_obs_indices].T

    x_temp, y_temp = create_training_data(fmat, lmat, nb + 1, nfeat, lookback)

    if n == 0:
        xtrain = x_temp
        ytrain = y_temp
    else:
        xtrain = np.vstack((xtrain, x_temp))
        ytrain = np.vstack((ytrain, y_temp))

# 4) Normalize data
p, q, r = xtrain.shape
data2d = xtrain.reshape(p * q, r)

scaler_in = MinMaxScaler(feature_range=(-1, 1))
data2d = scaler_in.fit_transform(data2d)
xtrain = data2d.reshape(p, q, r)

scaler_out = MinMaxScaler(feature_range=(-1, 1))
ytrain = scaler_out.fit_transform(ytrain)

# Train/validation split
xtrain, xvalid, ytrain, yvalid = train_test_split(
    xtrain, ytrain, test_size=0.2, shuffle=True
)

# 5) Build BiLSTM + GRU model
Training = True
if Training:
    input_layer = Input(shape=(lookback, r))
    bilstm_layer = Bidirectional(LSTM(80, return_sequences=True, activation='relu'))(input_layer)
    gru_layer = GRU(80, activation='relu')(bilstm_layer)
    dense_layer = Dense(80, activation='relu')(gru_layer)
    output_layer = Dense(ytrain.shape[1])(dense_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss=huber_loss, optimizer='adam', metrics=['mae'])

    csv_logger = CSVLogger('training.log')
    checkpoint = ModelCheckpoint('model_bilstm_gru.h5', monitor='val_loss',
                                 save_best_only=True, verbose=1, period=1)
    earlystop = EarlyStopping(monitor='val_loss', patience=50, verbose=1)

    history = model.fit(
        xtrain, ytrain,
        epochs=100,
        batch_size=128,
        validation_data=(xvalid, yvalid),
        callbacks=[checkpoint, csv_logger, earlystop]
    )

    # Evaluate on training data
    scores = model.evaluate(xtrain, ytrain, verbose=0)
    print(f"Training MAE: {scores[1]:.4f}")

# 6) Data assimilation with the trained BiLSTM+GRU
# Generate a trajectory with random initial error
uw_sim = np.zeros((ne, nt + 1))
mean_noise, var_noise = 0, 1.0e-2
std_noise = np.sqrt(var_noise)

u_init = utrue[:, 0] + np.random.normal(mean_noise, std_noise, ne)
uw_sim[:, 0] = u_init
for k in range(1, nt + 1):
    uw_sim[:, k] = rk4(ne, dt, uw_sim[:, k - 1], forcing)

# Load the trained model (with custom huber_loss)
model = load_model('model_bilstm_gru.h5', custom_objects={'huber_loss': huber_loss})

# Assimilation: each time we hit an observation
ulstm = np.zeros((ne, nt + 1))
ulstm[:, 0] = uw_sim[:, 0]

for k in range(1, nt + 1):
    # Forecast step
    state_next = rk4(ne, dt, ulstm[:, k - 1], forcing)
    ulstm[:, k] = state_next

    # Update step
    if k % freq == 0:
        obs_index = k // freq
        if obs_index <= nb:
            # Prepare input feature: current state + observed values
            feat = np.hstack((ulstm[:, k], uobs[oin, obs_index]))
            feat_sc = scaler_in.transform(feat.reshape(1, -1))
            feat_sc = feat_sc.reshape(1, lookback, nfeat)

            # Predict correction
            corr_sc = model.predict(feat_sc)
            corr = scaler_out.inverse_transform(corr_sc)

            # Apply correction
            ulstm[:, k] += corr.ravel()

# 7) Save results to CSV
t = np.linspace(0, tmax, nt + 1)
df_t = pd.DataFrame(t, columns=['t'])
df_t.to_csv('t.csv', index=False)

df_utrue = pd.DataFrame(utrue.T)
df_utrue.to_csv('utrue.csv', index=False)

df_uobs = pd.DataFrame(uobs.T)
df_uobs.to_csv('uobs.csv', index=False)

df_ulstm = pd.DataFrame(ulstm.T)
df_ulstm.to_csv('ulstm.csv', index=False)

print("CSV files saved successfully.")

# 8) Error metrics
mae_v1 = mean_absolute_error(utrue, ulstm)
mse_v1 = mean_squared_error(utrue, ulstm)
rmse_v1 = rmse(utrue, ulstm)
r2_v1 = r2_score(utrue, ulstm)

print("=== BiLSTM+GRU Assimilation Metrics ===")
print(f"MAE : {mae_v1:.4f}")
print(f"MSE : {mse_v1:.4f}")
print(f"RMSE: {rmse_v1:.4f}")
print(f"R²  : {r2_v1:.4f}")

print("\nProcess completed.")
