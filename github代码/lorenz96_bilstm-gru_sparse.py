#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 22:15:00 2020

@author: suraj
"""

import numpy as np
from numpy.random import seed
seed(1)
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt
import time as tm
import matplotlib.ticker as ticker

font = {'family' : 'Times New Roman',
        'size'   : 14}
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

#%%
def rhs(ne,u,fr):
    v = np.zeros(ne+3)
    v[2:ne+2] = u
    v[1] = v[ne+1]
    v[0] = v[ne]
    v[ne+2] = v[2]

    r = np.zeros(ne)

#    for i in range(2,ne+2):
#        r[i-2] = v[i-1]*(v[i+1] - v[i-2]) - v[i] + fr

    r = v[1:ne+1]*(v[3:ne+3] - v[0:ne]) - v[2:ne+2] + fr

    return r


def rk4(ne,dt,u,fr):
    r1 = rhs(ne,u,fr)
    k1 = dt*r1

    r2 = rhs(ne,u+0.5*k1,fr)
    k2 = dt*r2

    r3 = rhs(ne,u+0.5*k2,fr)
    k3 = dt*r3

    r4 = rhs(ne,u+k3,fr)
    k4 = dt*r4

    un = u + (k1 + 2.0*(k2 + k3) + k4)/6.0

    return un

#%%
ne = 40

dt = 0.005
tmax = 10.0
tini = 5.0
ns = int(tini/dt)
nt = int(tmax/dt)
fr = 8.0
nf = 10         # frequency of observation
nb = int(nt/nf) # number of observation time

u = np.zeros(ne)
utrue = np.zeros((ne,nt+1))
uinit = np.zeros((ne,ns+1))

#-----------------------------------------------------------------------------#
# generate true solution trajectory
#-----------------------------------------------------------------------------#
ti = np.linspace(-tini,0,ns+1)
t = np.linspace(0,tmax,nt+1)
tobs = np.linspace(0,tmax,nb+1)
x = np.linspace(1,ne,ne)

X,T = np.meshgrid(x,t,indexing='ij')
Xi,Ti = np.meshgrid(x,ti,indexing='ij')

u[:] = fr
u[int(ne/2)-1] = fr + 0.01
uinit[:,0] = u

# generate initial condition at t = 0
for k in range(1,ns+1):
    un = rk4(ne,dt,u,fr)
    uinit[:,k] = un
    u = np.copy(un)

# assign inital condition
u = uinit[:,-1]
utrue[:,0] = uinit[:,-1]

# generate true forward solution
for k in range(1,nt+1):
    un = rk4(ne,dt,u,fr)
    utrue[:,k] = un
    u = np.copy(un)

#%%
'''
fort = np.loadtxt('true_trajectory.plt',skiprows=1)
u1p = utrue[19,:]
u1f = fort[500:,2]
aa = u1p - u1f

field = np.loadtxt('true_field.plt',skiprows=2)
ufort = field[:,2].reshape(ns+nt+1,ne,order='f')

ufort_i = ufort[:501,:].T
ufort_e = ufort[500:,:].T

aa = utrue - ufort_e
'''
#%%
vmin = -10
vmax = 10
fig, ax = plt.subplots(2,1,figsize=(8,4))
cs = ax[0].contourf(Ti,Xi,uinit,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(uinit)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[0],ticks=np.linspace(vmin, vmax, 6))

#cs = ax[1].contourf(T,X,ufort_e,cmap='jet',vmin=-vmin,vmax=vmax)
#m = plt.cm.ScalarMappable(cmap='jet')
#m.set_array(uinit)
#m.set_clim(vmin, vmax)
#fig.colorbar(m,ax=ax[0],ticks=np.linspace(vmin, vmax, 6))

cs = ax[1].contourf(T,X,utrue,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(utrue)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[1],ticks=np.linspace(vmin, vmax, 6))
fig.savefig('output.png', dpi=600)

fig.tight_layout()
plt.show()

#%%
#-----------------------------------------------------------------------------#
# generate observations
#-----------------------------------------------------------------------------#
mean = 0.0
sd2 = 1.0e-2 # added noise (variance)
sd1 = np.sqrt(sd2) # added noise (standard deviation)

oib = [nf*k for k in range(nb+1)]

uobs = utrue[:,oib] + np.random.normal(mean,sd1,[ne,nb+1])

#-----------------------------------------------------------------------------#
# generate erroneous soltions trajectory
#-----------------------------------------------------------------------------#
uw = np.zeros((ne,nt+1))
k = 0
si2 = 1.0e-2
si1 = np.sqrt(si2)

u = utrue[:,0] + np.random.normal(mean,si1,ne)
uw[:,0] = u

for k in range(1,nt+1):
    un = rk4(ne,dt,u,fr)
    uw[:,k] = un
    u = np.copy(un)

#%%
#-----------------------------------------------------------------------------#
# EnKF model
#-----------------------------------------------------------------------------#

# number of observation vector
me = 4
freq = int(ne/me)
oin = [freq*i-1 for i in range(1,me+1)]

# number of ensemble
npe = 400
cn = 1.0/np.sqrt(npe-1)

z = np.zeros((me,nb+1))
zf = np.zeros((me,npe,nb+1))
DhX = np.zeros((me,npe))
DhXm = np.zeros(me)

ua = np.zeros((ne,nt+1)) # mean analyssi solution (to store)
uf = np.zeros(ne)        # mean forecast
sc = np.zeros((ne,npe))   # square-root of the covariance matrix
ue = np.zeros((ne,npe,nt+1)) # all ensambles
ph = np.zeros((ne,me))

km = np.zeros((ne,me))
kmd = np.zeros((ne,npe))

cc = np.zeros((me,me))
ci = np.zeros((me,me))

for k in range(nb+1):
    z[:,k] = uobs[oin,k]
    for n in range(npe):
        zf[:,n,k] = z[:,k] + np.random.normal(mean,sd1,me)
# Add code to compute and print MAE, MSE, RMSE, and R²
#-----------------------------------------------------
# Compute error metrics between EnKF results and true solution
MAE = np.mean(np.abs(utrue - ua))
MSE = np.mean((utrue - ua)**2)
RMSE = np.sqrt(MSE)
SStot = np.sum((utrue - np.mean(utrue))**2)
SSres = np.sum((utrue - ua)**2)
R2 = (1 - (SSres / SStot))**2

print("EnKF Assimilation Performance Metrics:")
print(f"MAE: {MAE}")
print(f"MSE: {MSE}")
print(f"RMSE: {RMSE}")
print(f"R²: {R2}")
#-----------------------------------------------------
# initial ensemble
k = 0
se2 = np.sqrt(sd2)
se1 = np.sqrt(se2)

for n in range(npe):
    ue[:,n,k] = uw[:,k] + np.random.normal(mean,sd1,ne)

ua[:,k] = np.sum(ue[:,:,k],axis=1)
ua[:,k] = ua[:,k]/npe

kobs = 1

# RK4 scheme
for k in range(1,nt+1):

    # forecast afor all ensemble fields
    for n in range(npe):
        u[:] = ue[:,n,k-1]
        un = rk4(ne,dt,u,fr)
        ue[:,n,k] = un[:] + np.random.normal(mean,se1,ne)

    if k == oib[kobs]:
        # compute mean of the forecast fields
        uf[:] = np.sum(ue[:,:,k],axis=1)
        uf[:] = uf[:]/npe

        # compute square-root of the covariance matrix
        for n in range(npe):
            sc[:,n] = cn*(ue[:,n,k] - uf[:])

        # compute DhXm data
        DhXm[:] = np.sum(ue[oin,:,k],axis=1)
        DhXm[:] = DhXm[:]/npe

        # compute DhM data
        for n in range(npe):
            DhX[:,n] = cn*(ue[oin,n,k] - DhXm[:])

        # R = sd2*I, observation m+atrix
        cc = DhX @ DhX.T

        for i in range(me):
            cc[i,i] = cc[i,i] + sd2

        ph = sc @ DhX.T

        ci = np.linalg.pinv(cc) # ci: inverse of cc matrix

        km = ph @ ci

        # analysis update
        kmd = km @ (zf[:,:,kobs] - ue[oin,:,k])
        ue[:,:,k] = ue[:,:,k] + kmd[:,:]

        kobs = kobs+1

    # mean analysis for plotting
    ua[:,k] = np.sum(ue[:,:,k],axis=1)
    ua[:,k] = ua[:,k]/npe

#%%
fig, ax = plt.subplots(3,1,sharex=True,figsize=(10,6))

n = [0,14,34]
for i in range(3):
    ax[i].plot(tobs,uobs[n[i],:],'ro', lw=3)
    ax[i].plot(t,utrue[n[i],:],'k-')
    ax[i].plot(t,uw[n[i],:],'b--')
    ax[i].plot(t,ua[n[i],:],'g-.')

    ax[i].set_xlim([0,tmax])
    ax[i].set_ylabel(r'$x_{'+str(n[i]+1)+'}$')

ax[i].set_xlabel(r'$t$')
line_labels = ['Observation','True','Wrong','EnKF']
plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=-0.2, ncol=4, labelspacing=0.)
fig.tight_layout()
plt.show()
fig.savefig('m_'+str(me)+'.pdf')

#%%
import numpy as np
import matplotlib.pyplot as plt

# 假设已经有了以下数据（您需要根据实际情况提供这些数据）
# T: 时间，X: 空间坐标，utrue: 真实数据，ua: EnKF同化结果
# 这些数据应该是适当维度的 NumPy 数组

vmin = -10
vmax = 10

# 创建3行1列的子图（原始数据，EnKF同化结果，EnKF同化误差）
fig, ax = plt.subplots(3, 1, figsize=(8, 10))

# 绘制真实值的等高线图
cs = ax[0].contourf(T, X, utrue, cmap='jet', vmin=vmin, vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(utrue)
m.set_clim(vmin, vmax)
fig.colorbar(m, ax=ax[0], ticks=np.linspace(vmin, vmax, 6))
ax[0].set_title('True')

# 绘制EnKF同化结果的等高线图
cs = ax[1].contourf(T, X, ua, cmap='jet', vmin=vmin, vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(ua)
m.set_clim(vmin, vmax)
fig.colorbar(m, ax=ax[1], ticks=np.linspace(vmin, vmax, 6))
ax[1].set_title('EnKF')

# 计算并绘制EnKF同化误差的等高线图
error = utrue - ua
cs = ax[2].contourf(T, X, error, cmap='jet', vmin=vmin, vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(error)
m.set_clim(vmin, vmax)
fig.colorbar(m, ax=ax[2], ticks=np.linspace(vmin, vmax, 6))
ax[2].set_title('EnKF Error')

# 自动调整布局，避免子图重叠
fig.tight_layout()

# 显示图像
plt.show()

# 保存为PDF文件，设置分辨率为600 DPI
fig.savefig('f_' + str(me) + '.pdf', dpi=600)




#%%
uwe = np.zeros((ne,npe,nt+1)) # all ensambles
k = 0
for n in range(npe):
    uwe[:,n,k] = uw[:,k] + np.random.normal(mean,sd1,ne)

for k in range(1,nt+1):
    for n in range(npe):
        u[:] = uwe[:,n,k-1]
        un = rk4(ne,dt,u,fr)
        uwe[:,n,k] = un[:]

np.savez('lstm_data_sparse.npz',utrue=utrue, uobs=uobs, uwe = uwe)

fig, ax = plt.subplots(3,1,sharex=True,figsize=(10,6))

n = [0,19,39]
for i in range(3):
    ax[i].plot(tobs,uobs[n[i],:],'ro',lw=3)
    ax[i].plot(t,utrue[n[i],:],'k-')
    ax[i].plot(t,uw[n[i],:],'b--')
    ax[i].plot(t,ua[n[i],:],'g-.')
    ax[i].plot(t,uwe[n[i],0,:],'m--')

    ax[i].set_xlim([0,tmax])
    ax[i].set_ylabel(r'$x_{'+str(n[i]+1)+'}$')

ax[i].set_xlabel(r'$t$')
line_labels = ['Observation','True','Wrong','EnKF','LSTM Error']
plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=-0.2, ncol=5, labelspacing=0.)
fig.tight_layout()
plt.show()
fig.savefig('m_'+str(me)+'.pdf')
