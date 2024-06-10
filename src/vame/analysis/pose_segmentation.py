#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational Animal Motion Embedding 1.0-alpha Toolbox
© K. Luxem & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0
"""

import os
import tqdm
import torch
import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple

from hmmlearn import hmm
from sklearn.cluster import KMeans

from vame.util.auxiliary import read_config
from vame.model.rnn_model import RNN_VAE


def load_model(cfg: dict, model_name: str, fixed: bool) -> RNN_VAE:
    """Load the VAME model.

    Args:
        cfg (dict): Configuration dictionary.
        model_name (str): Name of the model.
        fixed (bool): Fixed or variable length sequences.

    Returns:
        RNN_VAE: Loaded VAME model.
    """
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        pass
    else:
        torch.device("cpu")

    # load Model
    ZDIMS = cfg['zdims']
    FUTURE_DECODER = cfg['prediction_decoder']
    TEMPORAL_WINDOW = cfg['time_window']*2
    FUTURE_STEPS = cfg['prediction_steps']
    NUM_FEATURES = cfg['num_features']
    if fixed == False:
        NUM_FEATURES = NUM_FEATURES - 2
    hidden_size_layer_1 = cfg['hidden_size_layer_1']
    hidden_size_layer_2 = cfg['hidden_size_layer_2']
    hidden_size_rec = cfg['hidden_size_rec']
    hidden_size_pred = cfg['hidden_size_pred']
    dropout_encoder = cfg['dropout_encoder']
    dropout_rec = cfg['dropout_rec']
    dropout_pred = cfg['dropout_pred']
    softplus = cfg['softplus']


    if use_gpu:
        model = RNN_VAE(TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES,FUTURE_DECODER,FUTURE_STEPS, hidden_size_layer_1,
                                hidden_size_layer_2, hidden_size_rec, hidden_size_pred, dropout_encoder,
                                dropout_rec, dropout_pred, softplus).cuda()
    else:
        model = RNN_VAE(TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES,FUTURE_DECODER,FUTURE_STEPS, hidden_size_layer_1,
                                hidden_size_layer_2, hidden_size_rec, hidden_size_pred, dropout_encoder,
                                dropout_rec, dropout_pred, softplus).to()

    model.load_state_dict(torch.load(os.path.join(cfg['project_path'],'model','best_model',model_name+'_'+cfg['Project']+'.pkl')))
    model.eval()

    return model


def embedd_latent_vectors(cfg: dict, files: List[str], model: RNN_VAE, fixed: bool) -> List[np.ndarray]:
    """Embed latent vectors for the given files using the VAME model.

    Args:
        cfg (dict): Configuration dictionary.
        files (List[str]): List of files names.
        model (RNN_VAE): VAME model.
        fixed (bool): Whether the model is fixed.

    Returns:
        List[np.ndarray]: List of latent vectors for each file.
    """
    project_path = cfg['project_path']
    temp_win = cfg['time_window']
    num_features = cfg['num_features']
    if fixed == False:
        num_features = num_features - 2

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        pass
    else:
        torch.device("cpu")

    latent_vector_files = []

    for file in files:
        print('Embedding of latent vector for file %s' %file)
        data = np.load(os.path.join(project_path,'data',file,file+'-PE-seq-clean.npy'))
        latent_vector_list = []
        with torch.no_grad():
            for i in tqdm.tqdm(range(data.shape[1] - temp_win)):
            # for i in tqdm.tqdm(range(10000)):
                data_sample_np = data[:,i:temp_win+i].T
                data_sample_np = np.reshape(data_sample_np, (1, temp_win, num_features))
                if use_gpu:
                    h_n = model.encoder(torch.from_numpy(data_sample_np).type('torch.FloatTensor').cuda())
                else:
                    h_n = model.encoder(torch.from_numpy(data_sample_np).type('torch.FloatTensor').to())
                mu, _, _ = model.lmbda(h_n)
                latent_vector_list.append(mu.cpu().data.numpy())

        latent_vector = np.concatenate(latent_vector_list, axis=0)
        latent_vector_files.append(latent_vector)

    return latent_vector_files


def consecutive(data: np.ndarray, stepsize: int = 1) -> List[np.ndarray]:
    """Find consecutive sequences in the data array.

    Args:
        data (np.ndarray): Input array.
        stepsize (int, optional): Step size. Defaults to 1.

    Returns:
        List[np.ndarray]: List of consecutive sequences.
    """
    data = data[:]
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


def get_motif_usage(label: np.ndarray) -> np.ndarray:
    """Compute motif usage from the label array.

    Args:
        label (np.ndarray): Label array.

    Returns:
        np.ndarray: Array of motif usage counts.
    """
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


def same_parametrization(
    cfg: dict,
    files: List[str],
    latent_vector_files: List[np.ndarray],
    states: int,
    parametrization: str
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Apply the same parametrization to all animals.

    Args:
        cfg (dict): Configuration dictionary.
        files (List[str]): List of file names.
        latent_vector_files (List[np.ndarray]): List of latent vector arrays.
        states (int): Number of states.
        parametrization (str): parametrization method.

    Returns:
        Tuple: Tuple of labels, cluster centers, and motif usages.
    """
    random_state = cfg['random_state_kmeans']
    n_init = cfg['n_init_kmeans']

    labels = []
    cluster_centers = []
    motif_usages = []

    latent_vector_cat = np.concatenate(latent_vector_files, axis=0)

    if parametrization == "kmeans":
        print("Using kmeans as parametrization!")
        kmeans = KMeans(init='k-means++', n_clusters=states, random_state=42, n_init=20).fit(latent_vector_cat)
        clust_center = kmeans.cluster_centers_
        label = kmeans.predict(latent_vector_cat)

    elif parametrization == "hmm":
        if cfg['hmm_trained'] == False:
            print("Using a HMM as parametrization!")
            hmm_model = hmm.GaussianHMM(n_components=states, covariance_type="full", n_iter=100)
            hmm_model.fit(latent_vector_cat)
            label = hmm_model.predict(latent_vector_cat)
            save_data = os.path.join(cfg['project_path'], "results", "")
            with open(save_data+"hmm_trained.pkl", "wb") as file: pickle.dump(hmm_model, file)
        else:
            print("Using a pretrained HMM as parametrization!")
            save_data = os.path.join(cfg['project_path'], "results", "")
            with open(save_data+"hmm_trained.pkl", "rb") as file:
                hmm_model = pickle.load(file)
            label = hmm_model.predict(latent_vector_cat)

    idx = 0
    for i, file in enumerate(files):
        file_len = latent_vector_files[i].shape[0]
        labels.append(label[idx:idx+file_len])
        if parametrization == "kmeans":
            cluster_centers.append(clust_center)

        motif_usage = get_motif_usage(label[idx:idx+file_len])
        motif_usages.append(motif_usage)
        idx += file_len

    return labels, cluster_centers, motif_usages


def individual_parametrization(
    cfg: dict,
    files: List[str],
    latent_vector_files: List[np.ndarray],
    cluster: int
) -> Tuple:
    """Apply individual parametrization to each animal.

    Args:
        cfg (dict): Configuration dictionary.
        files (List[str]): List of file names.
        latent_vector_files (List[np.ndarray]): List of latent vector arrays.
        cluster (int): Number of clusters.

    Returns:
        Tuple: Tuple of labels, cluster centers, and motif usages.
    """
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


def pose_segmentation(config: str) -> None:
    """Perform pose segmentation using the VAME model.

    Args:
        config (str): Path to the configuration file.

    Returns:
        None
    """
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    legacy = cfg['legacy']
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    fixed = cfg['egocentric_data']
    parametrization = cfg['parametrization']

    print('Pose segmentation for VAME model: %s \n' %model_name)

    if legacy == True:
        from vame.legacy.segment_behavior import behavior_segmentation
        behavior_segmentation(config, model_name=model_name, cluster_method='kmeans', n_cluster=n_cluster)

    else:
        ind_param = cfg['individual_parametrization']

        for folders in cfg['video_sets']:
            if not os.path.exists(os.path.join(cfg['project_path'],"results",folders,model_name,"")):
                os.mkdir(os.path.join(cfg['project_path'],"results",folders,model_name,""))

        files = []
        if cfg['all_data'] == 'No':
            all_flag = input("Do you want to qunatify your entire dataset? \n"
                              "If you only want to use a specific dataset type filename: \n"
                              "yes/no/filename ")
            file = all_flag

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
        # files.append("mouse-3-1")
        # file="mouse-3-1"

        use_gpu = torch.cuda.is_available()
        if use_gpu:
            print("Using CUDA")
            print('GPU active:',torch.cuda.is_available())
            print('GPU used:',torch.cuda.get_device_name(0))
        else:
            print("CUDA is not working! Attempting to use the CPU...")
            torch.device("cpu")

        if not os.path.exists(os.path.join(cfg['project_path'],"results",file,model_name, parametrization+'-'+str(n_cluster),"")):
            new = True
            # print("Hello1")
            model = load_model(cfg, model_name, fixed)
            latent_vectors = embedd_latent_vectors(cfg, files, model, fixed)

            if ind_param == False:
                print("For all animals the same parametrization of latent vectors is applied for %d cluster" %n_cluster)
                labels, cluster_center, motif_usages = same_parametrization(cfg, files, latent_vectors, n_cluster, parametrization)
            else:
                print("Individual parametrization of latent vectors for %d cluster" %n_cluster)
                labels, cluster_center, motif_usages = individual_parametrization(cfg, files, latent_vectors, n_cluster)

        else:
            print('\n'
                  'For model %s a latent vector embedding already exists. \n'
                  'parametrization of latent vector with %d k-Means cluster' %(model_name, n_cluster))

            if os.path.exists(os.path.join(cfg['project_path'],"results",file,model_name, parametrization+'-'+str(n_cluster),"")):
                flag = input('WARNING: A parametrization for the chosen cluster size of the model already exists! \n'
                            'Do you want to continue? A new parametrization will be computed! (yes/no) ')
            else:
                flag = 'yes'

            if flag == 'yes':
                new = True
                latent_vectors = []
                for file in files:
                    path_to_latent_vector = os.path.join(cfg['project_path'],"results",file,model_name, parametrization+'-'+str(n_cluster),"")
                    latent_vector = np.load(os.path.join(path_to_latent_vector,'latent_vector_'+file+'.npy'))
                    latent_vectors.append(latent_vector)

                if ind_param == False:
                    print("For all animals the same parametrization of latent vectors is applied for %d cluster" %n_cluster)
                    labels, cluster_center, motif_usages = same_parametrization(cfg, files, latent_vectors, n_cluster, parametrization)
                else:
                    print("Individual parametrization of latent vectors for %d cluster" %n_cluster)
                    labels, cluster_center, motif_usages = individual_parametrization(cfg, files, latent_vectors, n_cluster)

            else:
                print('No new parametrization has been calculated.')
                new = False

        # print("Hello2")
        if new == True:
            for idx, file in enumerate(files):
                print(os.path.join(cfg['project_path'],"results",file,"",model_name,parametrization+'-'+str(n_cluster),""))
                if not os.path.exists(os.path.join(cfg['project_path'],"results",file,model_name,parametrization+'-'+str(n_cluster),"")):
                    try:
                        os.mkdir(os.path.join(cfg['project_path'],"results",file,"",model_name,parametrization+'-'+str(n_cluster),""))
                    except OSError as error:
                        print(error)

                save_data = os.path.join(cfg['project_path'],"results",file,model_name,parametrization+'-'+str(n_cluster),"")
                np.save(os.path.join(save_data,str(n_cluster)+'_' + parametrization + '_label_'+file), labels[idx])
                if parametrization=="kmeans":
                    np.save(os.path.join(save_data,'cluster_center_'+file), cluster_center[idx])
                np.save(os.path.join(save_data,'latent_vector_'+file), latent_vectors[idx])
                np.save(os.path.join(save_data,'motif_usage_'+file), motif_usages[idx])


            print("You succesfully extracted motifs with VAME! From here, you can proceed running vame.motif_videos() ")
                  # "to get an idea of the behavior captured by VAME. This will leave you with short snippets of certain movements."
                  # "To get the full picture of the spatiotemporal dynamic we recommend applying our community approach afterwards.")

