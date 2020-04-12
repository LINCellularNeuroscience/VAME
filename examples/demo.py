#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 16:50:23 2019

@author: luxemk
"""

import sys
sys.path.append('./vame')
from vame import VAME

# These paths have to be set manually 
working_directory = '/home/luxemk/Research/'
project='VAME-Project'
videos = ['/directory/to/your/video-1','/directory/to/your/video-2','...']
          
# Initialize your project
# Pose Estimation file has to be put manually into project folder "/VAME-Project/videos/pose-estimation/"
# Make sure the pose estimation files have the same name as the videos with an additional PE at the end
# example: video-1-PE.csv
config = VAME.init_new_project(project=project, videos=videos, pose_files=True, working_directory=working_directory)

config = '/home/luxemk/Research/VAME-Nov13-2019/config.yaml'
config = '/home/luxemk/Research/treadmill-Feb27-2020/config.yaml'

# If unrestrained experiment, align animal:
VAME.align(config)
VAME.create_trainingdata(config, model="temporal")

# If vame.align() is finished or retrained experiment:
VAME.spatial(config, model_name='resnet18_z16_cae', pretrained=False,  KL_START=0, ANNEALTIME=1)
VAME.spatial_model_features(config, model_name='resnet50_mse', save_in_folder=True, save_train=False)

VAME.temporal(config, model_name='MODEL_FUT', spatial_features=False, pretrained=False)
VAME.temporal_lstm(config, model_name='FULL_2D', spatial_features=False, pretrained=False)
VAME.temporal_aux(config, model_name='FULL', spatial_features=False, pretrained=False)
VAME.rnn_ablation(config, model_name='basic_v1_g_f_k', spatial_features=False, pretrained=False)

model_name='hid128_64_v2_z16_f15'

# Evaluate model
VAME.evaluate_model(config, model_name='FULL_3D', model='temporal', spatial_features=False)
 
# Quantify Behavior
VAME.behavior_quantification(config, model='temporal', model_name='FULL_2D_reproduce_night', 
                             cluster_method='kmeans', n_cluster=[30], spatial_features=False, file=None)

# Get behavioral transition matrix, model usage and graph
VAME.analyze_behavior(config, model='temporal', cluster_method='kmeans', n_cluster=30)


#VAME.ground_truth_validation(config, model='temporal',cluster_method='kmeans', n_cluster=24, select_dict=1)

################## TREADMILL ####################
VAME.align(config, egocentric=True)
VAME.create_trainingdata(config, model="temporal", egocentric=True)

VAME.temporal_lstm(config, model_name='treadmill_v1', spatial_features=False, pretrained=False)
VAME.temporal_aux(config, model_name='treadmill_v2_nk', spatial_features=False, pretrained=False)
VAME.evaluate_model(config, model_name='treadmill_v2_nk', model='temporal', spatial_features=False)

VAME.behavior_quantification(config, model='temporal', model_name='treadmill_v1', 
                             cluster_method='kmeans', n_cluster=[7], spatial_features=False, file=None)

VAME.analyze_behavior(config, model='temporal', cluster_method='kmeans', n_cluster=8)

# SANITY CHECKS
import h5py
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

from VAME.util.auxiliary import read_config
from VAME.util.video import read_hdf5

config_file = Path(config).resolve()
cfg = read_config(config_file)

weight = []
for i in range(60):
    new_weight = float(1/(1+np.exp(-0.9*(30+i-60))))
    weight.append(new_weight)

import torch
x = torch.randn(1,5,3)
xf =torch.flip(x,[0,1])
y = torch.rand(30,256,)

cat = torch.cat((x,x),2)

from scipy.stats import ortho_group as og
x = og.rvs(10)
I = x.T.dot(x)
n, m = 28, 5
H = np.random.randn(n, m)
u, s, vh = np.linalg.svd(H, full_matrices=False)
mat = u @ vh
print(mat @ mat.T)
np.trace(mat)

# Spectral relaxation example
Z = np.array([[4,6,8,3,1],
              [4,7,3,5,1],
              [5,6,8,9,1]], dtype=np.float64)
    
n, m = 256, 30
F = np.random.randn(n, m)
u, s, vh = np.linalg.svd(F, full_matrices=False)
F = u @ vh
print(F.T @ F)

u, s, vh = np.linalg.svd(Z, full_matrices=False)

F_up = F.T @ F
F_up = F_up * s[:m]

gram = Z.T @ Z
u, s_g, vh = np.linalg.svd(gram, full_matrices=False)

sqr = np.sqrt(s_g)

F = torch.tensor(s_g, requires_grad=True)
tr = torch.trace(F)
a = F @ F.T

Z = torch.empty(256,30, dtype=torch.double)
Z = Z.T
gram = Z.T @ Z

a,st,b = torch.svd(gram)

aa = F.T@gram@F

sq = torch.sqrt(F)

a = np.load(cfg['project_path']+'data/train/temporal/sequence_train_full.npy')
af = np.load(cfg['project_path']+'data/3D/mouse-4-2-3D.npy')
z = np.load(cfg['project_path']+'results/mouse-3-1/temporal_quantification/z_mouse-3-12stbiGRU2GRU_kmeans.npy')
label = np.load(cfg['project_path']+'results/mouse-3-1/temporal_quantification/kmeans-30_3D/30_km_label_mouse-3-1.npy')
file="mouse-1-1"
X = np.load(cfg['project_path']+'data/'+file+'/'+file+'-dlc-seq_inter_v3-3D.npy')

af = af[44:]
np.save(cfg['project_path']+'data/3D/mouse-4-2-3D.npy', af)
import scipy.signal
lbl = np.int64(scipy.signal.medfilt(label, 19))

# Testing different distances as cluster marker (cosine similarity, pairwise distance, KL)
# For KL transform z into probability distributions

x = X[:,100:130]
alpha = .2
num_samples = int(len(x.T) * alpha)

x_new = x.copy()
for i in range(num_samples):
    random_sample_1 = np.random.choice(len(x.T))
    random_sample_2 = np.random.choice(len(x.T))
    
    x_new[:, random_sample_1] = x_new[:, random_sample_2]

plt.plot(X.T, 'r')
plt.plot(af[44:], 'g')



mean = np.mean(X)
std = np.std(X)
X = (X-mean)/std

mean_3 = np.mean(af)
std_3 = np.std(af)
af = (af-mean_3) / std_3


# LOSSES
train_loss = np.load(cfg['project_path']+'/'+'model/temporal_model/losses'+'/train_losses'+model_name+'.npy')
mse_loss = np.load(cfg['project_path']+'/'+'model/temporal_model/losses'+'/mse_losses'+model_name+'.npy')
km_loss = np.load(cfg['project_path']+'/'+'model/temporal_model/losses'+'/kmeans_losses'+model_name+'.npy', allow_pickle=True)
kl_loss = np.load(cfg['project_path']+'/'+'model/temporal_model/losses'+'/kl_losses'+model_name+'.npy')
fut_loss = np.load(cfg['project_path']+'/'+'model/temporal_model/losses'+'/fut_losses'+model_name+'.npy')

import torch
km_losses = []
for i in range(203):
    km = km_loss[i].cpu().detach().numpy()
    km_losses.append(km)

mse = mse_loss #/ np.max(mse_loss)
fut = fut_loss #/ np.max(fut_loss)
km = km_losses #/ np.max(km_losses)
kl = kl_loss #/ np.max(kl_loss)


plt.plot(mse)
plt.plot(fut)
plt.plot(km)
plt.plot(kl)


fig, (ax1) = plt.subplots(1, 1)
fig.suptitle('Losses of our Model')
ax1.set(xlabel='Epochs', ylabel='loss [%]')
ax1.set_yscale("log")
ax1.plot(mse, label='MSE-Loss')
ax1.plot(fut, label='Prediction-Loss')
ax1.plot(km, label='KMeans-Loss')
ax1.plot(kl, label='KL-Loss')
ax1.legend()


def kmeans_clustering(context, n_clusters):    
    kmeans = KMeans(init='k-means++',n_clusters=n_clusters, random_state=42,n_init=15).fit(context)
    return kmeans.predict(context)

z = []
logger = []
log = 0
#logger.append(log)
for file in files:
    logger.append(log)
    path = cfg['project_path']+'results/'+file+'/temporal_quantification/z_'+file+'FULL_3D.npy'
    z_t = np.load(path)
    z.append(z_t)
    z_len = len(z_t)
    log += z_len
logger.append(log) 

z = np.concatenate(z)
lbl = kmeans_clustering(z, 30)

for idx, file in enumerate(files):
    labels = lbl[logger[idx]:logger[idx+1]:]
    np.save(cfg['project_path']+'results/'+file+'/temporal_quantification/kmeans-30_3D/'+file+'_label_3D.npy', labels)




