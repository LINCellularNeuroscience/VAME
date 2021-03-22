#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational Animal Motion Embedding 0.1 Toolbox
© K. Luxem & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0
"""

import os
import tqdm
import numpy as np
from pathlib import Path

import torch
import scipy.signal
from sklearn import mixture
from sklearn.cluster import KMeans

from vame.util.auxiliary import read_config
from vame.model.rnn_model import RNN_VAE, RNN_VAE_LEGACY


def load_model(cfg, model_name, legacy):
    # load Model
    ZDIMS = cfg['zdims']
    FUTURE_DECODER = cfg['prediction_decoder']
    TEMPORAL_WINDOW = cfg['time_window']*2
    FUTURE_STEPS = cfg['prediction_steps']
    NUM_FEATURES = cfg['num_features']
    if legacy == False:
        NUM_FEATURES = NUM_FEATURES - 2
    hidden_size_layer_1 = cfg['hidden_size_layer_1']
    hidden_size_layer_2 = cfg['hidden_size_layer_2']
    hidden_size_rec = cfg['hidden_size_rec']
    hidden_size_pred = cfg['hidden_size_pred']
    dropout_encoder = cfg['dropout_encoder']
    dropout_rec = cfg['dropout_rec']
    dropout_pred = cfg['dropout_pred']
     
    if legacy == False:
        print('Load model... ')
        model = RNN_VAE(TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES,FUTURE_DECODER,FUTURE_STEPS, hidden_size_layer_1, 
                                hidden_size_layer_2, hidden_size_rec, hidden_size_pred, dropout_encoder, 
                                dropout_rec, dropout_pred).cuda()
    else:
        print('ATTENTIION - legacy model... ')
        model = RNN_VAE_LEGACY(TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES,FUTURE_DECODER,FUTURE_STEPS, hidden_size_layer_1, 
                                hidden_size_layer_2, hidden_size_rec, hidden_size_pred, dropout_encoder, 
                                dropout_rec, dropout_pred).cuda()
    
    model.load_state_dict(torch.load(os.path.join(cfg['project_path'],'model','best_model',model_name+'_'+cfg['Project']+'.pkl')))
    model.eval()
    
    return model


def embedd_latent_vectors(cfg, files, model, legacy):
    project_path = cfg['project_path']
    temp_win = cfg['time_window']
    num_features = cfg['num_features']
    if legacy == False:
        num_features = num_features - 2
        
    latent_vector_files = [] 

    for file in files:
        print('Embedd latent vector for file %s' %file)
        data = np.load(os.path.join(project_path,'data',file,file+'-PE-seq-clean.npy'))
        latent_vector_list = []
        with torch.no_grad(): 
            for i in tqdm.tqdm(range(data.shape[1] - temp_win)):
                data_sample_np = data[:,i:temp_win+i].T
                data_sample_np = np.reshape(data_sample_np, (1, temp_win, num_features))
                h_n = model.encoder(torch.from_numpy(data_sample_np).type('torch.FloatTensor').cuda())
                _, mu, _ = model.lmbda(h_n)
                latent_vector_list.append(mu.cpu().data.numpy())
        
        latent_vector = np.concatenate(latent_vector_list, axis=0)
        latent_vector_files.append(latent_vector)
        
    return latent_vector_files


def consecutive(data, stepsize=1):
    data = data[:]
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


def get_motif_usage(label):
    motif_usage = np.unique(label, return_counts=True)
    cons = consecutive(motif_usage[0])
    if len(cons) != 1:
        usage_list = list(motif_usage[1])
        for i in range(len(cons)-1):
            a = cons[i+1][0]
            b = cons[i][-1]
            d = (a-b)-1
            for j in range(1,d+1):
                index = cons[i][-1]+j
                usage_list.insert(index,0)
        usage = np.array(usage_list)
        motif_usage = usage
    else:
        motif_usage = motif_usage[1]
    
    return motif_usage


def same_parameterization(cfg, files, latent_vector_files, cluster):
    random_state = cfg['random_state_kmeans']
    n_init = cfg['n_init_kmeans']
    
    labels = []
    cluster_centers = []
    motif_usages = []
    latent_vector_cat = np.concatenate(latent_vector_files, axis=0)
    kmeans = KMeans(init='k-means++', n_clusters=cluster, random_state=random_state, n_init=n_init).fit(latent_vector_cat)
    clust_center = kmeans.cluster_centers_
    label = kmeans.predict(latent_vector_cat)
    
    idx = 0
    for i, file in enumerate(files):
        file_len = latent_vector_files[i].shape[0]
        labels.append(label[idx:idx+file_len])
        cluster_centers.append(clust_center)
        
        motif_usage = get_motif_usage(label[idx:idx+file_len])
        motif_usages.append(motif_usage)
        idx += file_len
    
    return labels, cluster_centers, motif_usages
    

def individual_parameterization(cfg, files, latent_vector_files, cluster):
    random_state = cfg['random_state_kmeans: ']
    n_init = cfg['n_init_kmeans']
    
    labels = []
    cluster_centers = []
    motif_usages = []
    for i, file in enumerate(files):
        print(file)
        kmeans = KMeans(init='k-means++', n_clusters=cluster, random_state=random_state, n_init=n_init).fit(latent_vector_files[i])
        clust_center = kmeans.cluster_centers_
        label = kmeans.predict(latent_vector_files[i])  
        motif_usage = get_motif_usage(label)
        motif_usages.append(motif_usage)
        labels.append(label)
        cluster_centers.append(clust_center)
    
    return labels, cluster_centers, motif_usages


def pose_segmentation(config):

    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    legacy = cfg['legacy']
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    
    print('Pose segmentation for VAME model: %s \n' %model_name)
    
    if legacy == True:
        from segment_behavior import behavior_segmentation
        behavior_segmentation(config, model_name=model_name, cluster_method='kmeans', n_cluster=n_cluster)
        
    if legacy == False:
        ind_param = cfg['individual_parameterization']
        
        for folders in cfg['video_sets']:
            if not os.path.exists(os.path.join(cfg['project_path'],"results",folders,model_name,"")):
                os.mkdir(os.path.join(cfg['project_path'],"results",folders,model_name,""))
    
        files = []
        if cfg['all_data'] == 'No':
            all_flag = input("Do you want to qunatify your entire dataset? \n"
                             "If you only want to use a specific dataset type filename: \n"
                             "yes/no/filename ")
        else:
            all_flag = 'yes'
    
        if all_flag == 'yes' or all_flag == 'Yes':
            for file in cfg['video_sets']:
                files.append(file)
        elif all_flag == 'no' or all_flag == 'No':
            for file in cfg['video_sets']:
                use_file = input("Do you want to quantify " + file + "? yes/no: ")
                if use_file == 'yes':
                    files.append(file)
                if use_file == 'no':
                    continue
        else:
            files.append(all_flag)
    
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            print("Using CUDA")
            print('GPU active:',torch.cuda.is_available())
            print('GPU used:',torch.cuda.get_device_name(0))
        else:
            print("CUDA is not working! Attempting to use the CPU...")
            torch.device("cpu")
        
        folder = os.path.dirname(os.path.join(cfg['project_path'],"results",file,model_name,""))
        if not os.listdir(folder):
            print(os.path.join(cfg['project_path'],"results",file,model_name,""))
            model = load_model(cfg, model_name, legacy)
            latent_vectors = embedd_latent_vectors(cfg, files, model, legacy)

            if ind_param == False:
                print("For all animals the same k-Means parameterization of latent vectors is applied for %d cluster" %n_cluster)
                labels, cluster_center, motif_usages = same_parameterization(cfg, files, latent_vectors, n_cluster)
            else:
                print("Individual k-Means parameterization of latent vectors for %d cluster" %n_cluster)
                labels, cluster_center, motif_usages = individual_parameterization(cfg, files, latent_vectors, n_cluster)
            
            print(files)
            for idx, file in enumerate(files):
                print(os.path.join(cfg['project_path'],"results",file,"",model_name,'kmeans-'+str(n_cluster),""))
                if not os.path.exists(os.path.join(cfg['project_path'],"results",file,model_name,'kmeans-'+str(n_cluster),"")):                    
                    try:
                        os.mkdir(os.path.join(cfg['project_path'],"results",file,"",model_name,'kmeans-'+str(n_cluster),""))
                    except OSError as error:
                        print(error)                    
                    
                save_data = os.path.join(cfg['project_path'],"results",file,model_name,'kmeans-'+str(n_cluster),"")
                np.save(os.path.join(save_data,str(n_cluster)+'_km_label_'+file), labels[idx])
                np.save(os.path.join(save_data,'cluster_center_'+file), cluster_center[idx])
                np.save(os.path.join(save_data,'latent_vector_'+file), latent_vectors[idx])
                np.save(os.path.join(save_data,'motif_usage_'+file), motif_usages[idx])
            
        else:
            print('\n'
                  'For model %s a latent vector embedding already exists. \n' 
                  'Parameterization of latent vector with %d k-Means cluster \n' %(model_name, n_cluster))
            
            if os.path.exists(os.path.join(cfg['project_path'],"results",file,model_name,'kmeans-'+str(n_cluster),"")):
                flag = input('WARNING: A parameterization for the chosen cluster size of the model already exists! \n'
                             'Do you want to continue? A new k-Means assignment will be computed! (yes/no) ')
            else:
                flag = 'yes'
            
            if flag == 'yes':
                path_to_latent_vector = os.listdir(folder)[0]
                latent_vectors = []
                for file in files:
                    latent_vector = np.load(os.path.join(cfg['project_path'],"results",file,model_name,path_to_latent_vector,'latent_vector_'+file+'.npy'))
                    latent_vectors.append(latent_vector)
                    
                if ind_param == False:
                    print("For all animals the same k-Means parameterization of latent vectors is applied for %d cluster" %n_cluster)
                    labels, cluster_center, motif_usages = same_parameterization(cfg, files, latent_vectors, n_cluster)
                else:
                    print("Individual k-Means parameterization of latent vectors for %d cluster" %n_cluster)
                    labels, cluster_center, motif_usages = individual_parameterization(cfg, files, latent_vectors, n_cluster)
                
                
                for idx, file in enumerate(files):
                    print(os.path.join(cfg['project_path'],"results",file,"",model_name,'kmeans-'+str(n_cluster),""))
                    if not os.path.exists(os.path.join(cfg['project_path'],"results",file,model_name,'kmeans-'+str(n_cluster),"")):                    
                        try:
                            os.mkdir(os.path.join(cfg['project_path'],"results",file,"",model_name,'kmeans-'+str(n_cluster),""))
                        except OSError as error:
                            print(error)   
                        
                    save_data = os.path.join(cfg['project_path'],"results",file,model_name,'kmeans-'+str(n_cluster),"")
                    np.save(os.path.join(save_data,str(n_cluster)+'_km_label_'+file), labels[idx])
                    np.save(os.path.join(save_data,'cluster_center_'+file), cluster_center[idx])
                    np.save(os.path.join(save_data,'latent_vector_'+file), latent_vectors[idx])
                    np.save(os.path.join(save_data,'motif_usage_'+file), motif_usages[idx])
            else:
                print('No new parameterization has been calculated.')
            
        
        print("You succesfully extracted motifs with VAME! From here, you can proceed running vame.motif_videos() "
              "to get an idea of the behavior captured by VAME. This will leave you with short snippets of certain movements."
              "To get the full picture of the spatiotemporal dynamic we recommend applying our community approach afterwards.")
        
        
        
        
        
        
        
        
        
        