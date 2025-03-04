import numpy as np
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import keras.backend as K
K.set_floatx('float64')

from keras import Input, Model
from keras.layers import Dense, GRU, Bidirectional, LSTM
from keras.models import load_model
from keras.losses import Huber
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------- Lorenz-96 Functions ---------------------
def rhs(ne, u, forcing):
    """
    Lorenz-96 right-hand side function.
    ne: number of variables in the L96 system (state dimension).
    u: 1D array of the state variables.
    forcing: forcing term (F).
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
    4th-order Runge-Kutta integration.
    ne: number of variables.
    dt: time step.
    u: current state (1D array).
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

    un = u + (k1 + 2.0 * (k2 + k3) + k4) / 6.0
    return un

# --------------------- BiLSTM-GRU Data Preparation ---------------------
def create_training_data_lstm(features, labels, m, n, lookback):
    """
    Convert a feature matrix of shape (m x n) into a 3D input of shape
    (m - lookback + 1, lookback, n), aligning the labels accordingly.
    """
    y_train = [labels[i, :] for i in range(m)]
    y_train = np.array(y_train)

    x_train = np.zeros((m - lookback + 1, lookback, n))
    for i in range(m - lookback + 1):
        temp = features[i, :]
        for j in range(1, lookback):
            temp = np.vstack((temp, features[i + j, :]))
        x_train[i, :, :] = temp

    return x_train, y_train

def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

# --------------------- Main Program ---------------------
# 1) Set parameters
ne = 40         # Dimension for Lorenz-96
npe = 400       # Number of ensemble members / samples
forcing = 10.0  # Lorenz-96 forcing term
dt = 0.005      # Time step
tmax = 10.0
nt = int(tmax / dt)

lookback = 1
me = 4          # Number of observations
freq = ne // me # Spacing for observed state variables
obs_idx = [freq * i - 1 for i in range(1, me + 1)]
nf = 10         # Observation frequency in time (every nf steps)
nb = nt // nf
time_obs_indices = [nf * k for k in range(nb + 1)]

# 2) Load data
data = np.load('lstm_data_sparse.npz')
utrue = data['utrue']  # shape: (ne, nt+1)
uobs = data['uobs']    # shape: (ne, nb+1)
uwe  = data['uwe']     # shape: (ne, npe, nt+1)

# 3) Prepare features & labels
nfeat = ne + me

for n in range(npe):
    feat_matrix = np.hstack((
        uwe[:, n, time_obs_indices].T,  # shape: (nb+1, ne)
        uobs[obs_idx, :].T              # shape: (nb+1, me)
    ))
    # Label: (utrue - uwe) at obs times
    label_matrix = (utrue[:, time_obs_indices].T -
                    uwe[:, n, time_obs_indices].T)   # shape: (nb+1, ne)

    x_temp, y_temp = create_training_data_lstm(
        feat_matrix, label_matrix, nb + 1, nfeat, lookback
    )

    if n == 0:
        xtrain = x_temp
        ytrain = y_temp
    else:
        xtrain = np.vstack((xtrain, x_temp))
        ytrain = np.vstack((ytrain, y_temp))

# 4) Data normalization
scaler_in = MinMaxScaler(feature_range=(-1, 1))
scaler_out = MinMaxScaler(feature_range=(-1, 1))

p, q, r = xtrain.shape  # p: number of samples, q: lookback, r: features
data2d = xtrain.reshape(p * q, r)
data2d = scaler_in.fit_transform(data2d)
xtrain = data2d.reshape(p, q, r)

ytrain = scaler_out.fit_transform(ytrain)

# 5) Train-test split
from sklearn.model_selection import train_test_split
xtrain, xvalid, ytrain, yvalid = train_test_split(
    xtrain, ytrain, test_size=0.2, shuffle=True
)

# 6) Build and train BiLSTM+GRU model
from keras.layers import LSTM

input_layer = Input(shape=(lookback, r))
bilstm_layer = Bidirectional(
    LSTM(80, return_sequences=True, activation='relu')
)(input_layer)
gru_layer = GRU(80, activation='relu')(bilstm_layer)
dense_layer = Dense(80, activation='relu')(gru_layer)
output_layer = Dense(ytrain.shape[1])(dense_layer)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss=Huber(), optimizer='adam', metrics=['mean_squared_error'])

checkpoint = ModelCheckpoint('model_bilstm_gru.h5', monitor='val_loss',
                             save_best_only=True, verbose=1, period=1)
earlystop  = EarlyStopping(monitor='val_loss', patience=50, verbose=1)

history = model.fit(
    xtrain, ytrain,
    epochs=3000,
    batch_size=128,
    validation_data=(xvalid, yvalid),
    callbacks=[checkpoint, earlystop]
)

# 7) Deployment: data assimilation
# Generate a single trajectory with initial error
uw_assim = np.zeros((ne, nt + 1))
mean_error, sigma2 = 0, 1.0e-2
sigma = np.sqrt(sigma2)

# Perturb the initial true state
u0 = utrue[:, 0] + np.random.normal(mean_error, sigma, ne)
uw_assim[:, 0] = u0
for k in range(1, nt + 1):
    uw_assim[:, k] = rk4(ne, dt, uw_assim[:, k - 1], forcing)

# Assimilation
ulstm = np.zeros((ne, nt + 1))
ulstm[:, 0] = uw_assim[:, 0]

for k in range(1, nt + 1):
    # Forecast
    state_next = rk4(ne, dt, ulstm[:, k - 1], forcing)
    ulstm[:, k] = state_next

    # Observation update
    if k % nf == 0:
        obs_index = k // nf
        if obs_index <= nb:
            feat = np.hstack((ulstm[:, k], uobs[obs_idx, obs_index]))
            feat_scaled = scaler_in.transform(feat.reshape(1, -1))
            feat_scaled = feat_scaled.reshape(1, lookback, nfeat)

            # Predict correction
            corr_scaled = model.predict(feat_scaled)
            corr = scaler_out.inverse_transform(corr_scaled)

            # Apply correction
            ulstm[:, k] += corr.ravel()

# 8) Compute error metrics
mae_val  = mean_absolute_error(utrue, ulstm)
mse_val  = mean_squared_error(utrue, ulstm)
rmse_val = rmse(utrue, ulstm)
r2_val   = r2_score(utrue, ulstm)

print("=== BiLSTM + GRU Assimilation Metrics ===")
print(f"MAE : {mae_val:.4f}")
print(f"MSE : {mse_val:.4f}")
print(f"RMSE: {rmse_val:.4f}")
print(f"R²  : {r2_val:.4f}")

print("\nTraining and assimilation process finished.")
