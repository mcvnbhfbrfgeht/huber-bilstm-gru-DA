# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 17:41:10 2019
@author: Suraj
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras.backend as K
from keras import Input, Model
from keras.models import load_model
from keras.layers import Dense, LSTM, GRU, Bidirectional
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.losses import Huber
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------------------------------------------
# Reproducibility: set random seeds
# ----------------------------------------------------------------
np.random.seed(1)
import tensorflow as tf
tf.set_random_seed(2)

# ----------------------------------------------------------------
# Lorenz-96 and RK4
# ----------------------------------------------------------------
def rhs(ne, u, forcing):
    """
    Lorenz-96 right-hand side.
    ne: number of variables
    u:  state vector (1D)
    forcing: forcing term
    """
    v = np.zeros(ne + 3)
    v[2:ne + 2] = u
    v[1] = v[ne + 1]
    v[0] = v[ne]
    v[ne + 2] = v[2]
    return (
        v[1:ne + 1] * (v[3:ne + 3] - v[0:ne])
        - v[2:ne + 2]
        + forcing
    )

def rk4(ne, dt, u, forcing):
    """
    4th-order Runge-Kutta integration.
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

# ----------------------------------------------------------------
# Data preparation function
# ----------------------------------------------------------------
def create_training_data(features, labels, m, n, lookback):
    """
    Convert (m x n) features into (m-lookback+1, lookback, n).
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

# ----------------------------------------------------------------
# Main Program
# ----------------------------------------------------------------
ne = 40            # dimension of Lorenz-96
npe = 400          # ensemble size / sample count
forcing = 10.0     # Lorenz-96 forcing
dt = 0.005
tmax = 10.0
nt = int(tmax / dt)
lookback = 1

nf = 10            # observation frequency (time)
nb = nt // nf
obs_times = [nf * k for k in range(nb + 1)]
tobs = np.linspace(0, tmax, nb + 1)

t = np.linspace(0, tmax, nt + 1)
x_space = np.linspace(1, ne, ne)

# Load data (adjust file path if needed)
data = np.load('lstm_data_sparse.npz')
utrue = data['utrue']  # shape: (ne, nt+1)
uobs  = data['uobs']   # shape: (ne, nb+1)
uwe   = data['uwe']    # shape: (ne, npe, nt+1)

# Observed variables
me = 4
freq = ne // me
oin = [freq * i - 1 for i in range(1, me + 1)]

# Let's use ne + me as input features
nfeat = ne + me

# Build training set
for n in range(npe):
    # features: (uwe at obs times) + (uobs at those indices)
    f_mat = np.hstack((uwe[:, n, obs_times].T, uobs[oin, :].T))  # shape: (nb+1, ne+me)
    # labels: true - prior
    l_mat = utrue[:, obs_times].T - uwe[:, n, obs_times].T

    x_temp, y_temp = create_training_data(f_mat, l_mat, nb + 1, nfeat, lookback)

    if n == 0:
        xtrain = x_temp
        ytrain = y_temp
    else:
        xtrain = np.vstack((xtrain, x_temp))
        ytrain = np.vstack((ytrain, y_temp))

# Normalize data
p, q, r = xtrain.shape  # p: samples, q: lookback, r: features
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

# -------------------------- Training ----------------------------
Training = True
if Training:
    input_layer = Input(shape=(lookback, r))
    # Replace the original RNN with BiLSTM + GRU
    bilstm_layer = Bidirectional(
        LSTM(80, return_sequences=True, activation='relu')
    )(input_layer)
    gru_layer = GRU(80, activation='relu')(bilstm_layer)
    dense_layer = Dense(80, activation='relu')(gru_layer)
    output_layer = Dense(ytrain.shape[1])(dense_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss=Huber(), optimizer='adam', metrics=['mean_squared_error'])

    csv_logger = CSVLogger('training.log')
    checkpoint = ModelCheckpoint(
        filepath='model_bilstm_gru.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        period=1
    )

    callbacks_list = [
        checkpoint,
        csv_logger,
        EarlyStopping(monitor='val_loss', patience=50, verbose=1)
    ]

    history = model.fit(
        xtrain, ytrain,
        epochs=3000,
        batch_size=128,
        callbacks=callbacks_list,
        validation_data=(xvalid, yvalid)
    )

    # Evaluate
    scores = model.evaluate(xtrain, ytrain, verbose=0)
    print(f"Training MSE: {scores[1]:.4f}")

# -------------------------- Deployment --------------------------
# Generate a trajectory with random initial error
uw_sim = np.zeros((ne, nt + 1))
np.random.seed(0)
mean_noise, var_noise = 0, 1.0e-2
std_noise = np.sqrt(var_noise)

u_init = utrue[:, 0] + np.random.normal(mean_noise, std_noise, ne)
uw_sim[:, 0] = u_init
for k in range(1, nt + 1):
    uw_sim[:, k] = rk4(ne, dt, uw_sim[:, k - 1], forcing)

# Load the trained model
model = load_model('model_bilstm_gru.h5', custom_objects={'Huber': Huber})

# Assimilate step by step
ulstm = np.zeros((ne, nt + 1))
ulstm[:, 0] = uw_sim[:, 0]

for k in range(1, nt + 1):
    forecast = rk4(ne, dt, ulstm[:, k - 1], forcing)
    ulstm[:, k] = forecast

    if k % freq == 0:
        obs_idx = k // freq
        if obs_idx <= nb:
            # Create features: current state + observed variables
            feat = np.hstack((ulstm[:, k], uobs[oin, obs_idx]))
            feat_scaled = scaler_in.transform(feat.reshape(1, -1))
            feat_scaled = feat_scaled.reshape(1, lookback, nfeat)

            # Predict correction
            corr_scaled = model.predict(feat_scaled)
            corr = scaler_out.inverse_transform(corr_scaled)

            # Apply correction
            ulstm[:, k] += corr.ravel()

# Save results
df_t = pd.DataFrame(t, columns=['t'])
df_ulstm = pd.DataFrame(ulstm.T)

df_t.to_csv('t.csv', index=False)
df_ulstm.to_csv('ulstm.csv', index=False)
print("CSV files saved.")

# Metrics
mae_val = mean_absolute_error(utrue, ulstm)
mse_val = mean_squared_error(utrue, ulstm)
rmse_val = rmse(utrue, ulstm)
r2_val  = r2_score(utrue, ulstm)

print("=== BiLSTM+GRU Metrics ===")
print(f"MAE : {mae_val:.4f}")
print(f"MSE : {mse_val:.4f}")
print(f"RMSE: {rmse_val:.4f}")
print(f"R²  : {r2_val:.4f}")

print("\nProcess completed.")
