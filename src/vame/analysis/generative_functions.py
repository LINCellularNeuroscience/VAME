"""
Variational Animal Motion Embedding 1.0-alpha Toolbox
© K. Luxem & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0
"""

import os

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from vame.schemas.states import GenerativeModelFunctionSchema, save_state
from vame.util.auxiliary import read_config
from vame.model.rnn_model import RNN_VAE
from vame.logging.logger import VameLogger

logger_config = VameLogger(__name__)
logger = logger_config.logger


def random_generative_samples_motif(
    cfg: dict,
    model: torch.nn.Module,
    latent_vector: np.ndarray,
    labels: np.ndarray,
    n_cluster: int
) -> None:
    """Generate random samples for motifs.

    Args:
        cfg (dict): Configuration dictionary.
        model (torch.nn.Module): PyTorch model.
        latent_vector (np.ndarray): Latent vectors.
        labels (np.ndarray): Labels.
        n_cluster (int): Number of clusters.

    Returns:
        None: Plot of generated samples.
    """
    logger.info('Generate random generative samples for motifs...')
    time_window = cfg['time_window']
    for j in range(n_cluster):

        inds=np.where(labels==j)
        motif_latents=latent_vector[inds[0],:]
        gm = GaussianMixture(n_components=10).fit(motif_latents)

        # draw sample from GMM
        density_sample = gm.sample(10)

        # generate image via model decoder
        tensor_sample = torch.from_numpy(density_sample[0]).type('torch.FloatTensor')
        if torch.cuda.is_available():
            tensor_sample = tensor_sample.cuda()
        else:
            tensor_sample = tensor_sample.cpu()

        decoder_inputs = tensor_sample.unsqueeze(2).repeat(1, 1, time_window)
        decoder_inputs = decoder_inputs.permute(0,2,1)

        image_sample = model.decoder(decoder_inputs, tensor_sample)
        recon_sample = image_sample.cpu().detach().numpy()

        fig, axs = plt.subplots(2,5)
        for i in range(5):
            axs[0,i].plot(recon_sample[i,...])
            axs[1,i].plot(recon_sample[i+5,...])
        plt.suptitle('Generated samples for motif '+str(j))
        return fig

def random_generative_samples(cfg: dict, model: torch.nn.Module, latent_vector: np.ndarray) -> None:
    """Generate random generative samples.

    Args:
        cfg (dict): Configuration dictionary.
        model (torch.nn.Module): PyTorch model.
        latent_vector (np.ndarray): Latent vectors.

    Returns:
        None
    """
    logger.info('Generate random generative samples...')
    # Latent sampling and generative model
    time_window = cfg['time_window']
    gm = GaussianMixture(n_components=10).fit(latent_vector)

    # draw sample from GMM
    density_sample = gm.sample(10)

    # generate image via model decoder
    tensor_sample = torch.from_numpy(density_sample[0]).type('torch.FloatTensor')
    if torch.cuda.is_available():
        tensor_sample = tensor_sample.cuda()
    else:
        tensor_sample = tensor_sample.cpu()

    decoder_inputs = tensor_sample.unsqueeze(2).repeat(1, 1, time_window)
    decoder_inputs = decoder_inputs.permute(0,2,1)

    image_sample = model.decoder(decoder_inputs, tensor_sample)
    recon_sample = image_sample.cpu().detach().numpy()

    fig, axs = plt.subplots(2,5)
    for i in range(5):
        axs[0,i].plot(recon_sample[i,...])
        axs[1,i].plot(recon_sample[i+5,...])
    plt.suptitle('Generated samples')
    return fig


def random_reconstruction_samples(cfg: dict, model: torch.nn.Module, latent_vector: np.ndarray) -> None:
    """Generate random reconstruction samples.

    Args:
        cfg (dict): Configuration dictionary.
        model (torch.nn.Module): PyTorch model to use.
        latent_vector (np.ndarray): Latent vectors.

    Returns:
        None
    """
    logger.info('Generate random reconstruction samples...')
    # random samples for reconstruction
    time_window = cfg['time_window']

    rnd = np.random.choice(latent_vector.shape[0], 10)
    tensor_sample = torch.from_numpy(latent_vector[rnd]).type('torch.FloatTensor')
    if torch.cuda.is_available():
        tensor_sample = tensor_sample.cuda()
    else:
        tensor_sample = tensor_sample.cpu()

    decoder_inputs = tensor_sample.unsqueeze(2).repeat(1, 1, time_window)
    decoder_inputs = decoder_inputs.permute(0,2,1)

    image_sample = model.decoder(decoder_inputs, tensor_sample)
    recon_sample = image_sample.cpu().detach().numpy()

    fig, axs = plt.subplots(2,5)
    for i in range(5):
        axs[0,i].plot(recon_sample[i,...])
        axs[1,i].plot(recon_sample[i+5,...])
    plt.suptitle('Reconstructed samples')
    return fig


def visualize_cluster_center(cfg: dict, model: torch.nn.Module, cluster_center: np.ndarray) -> None:
    """Visualize cluster centers.

    Args:
        cfg (dict): Configuration dictionary.
        model (torch.nn.Module): PyTorch model.
        cluster_center (np.ndarray): Cluster centers.

    Returns:
        None
    """
    #Cluster Center
    logger.info('Visualize cluster center...')
    time_window = cfg['time_window']
    animal_centers = cluster_center

    tensor_sample = torch.from_numpy(animal_centers).type('torch.FloatTensor')
    if torch.cuda.is_available():
        tensor_sample = tensor_sample.cuda()
    else:
        tensor_sample = tensor_sample.cpu()
    decoder_inputs = tensor_sample.unsqueeze(2).repeat(1, 1, time_window)
    decoder_inputs = decoder_inputs.permute(0,2,1)

    image_sample = model.decoder(decoder_inputs, tensor_sample)
    recon_sample = image_sample.cpu().detach().numpy()

    num = animal_centers.shape[0]
    b = int(np.ceil(num / 5))

    fig, axs = plt.subplots(5,b)
    idx = 0
    for k in range(5):
        for i in range(b):
            axs[k,i].plot(recon_sample[idx,...])
            axs[k,i].set_title("Cluster %d" %idx)
            idx +=1
    return fig


def load_model(cfg: dict, model_name: str) -> torch.nn.Module:
    """Load PyTorch model.

    Args:
        cfg (dict): Configuration dictionary.
        model_name (str): Name of the model.

    Returns:
        torch.nn.Module: Loaded PyTorch model.
    """
    ZDIMS = cfg['zdims']
    FUTURE_DECODER = cfg['prediction_decoder']
    TEMPORAL_WINDOW = cfg['time_window']*2
    FUTURE_STEPS = cfg['prediction_steps']

    NUM_FEATURES = cfg['num_features']
    NUM_FEATURES = NUM_FEATURES - 2

    hidden_size_layer_1 = cfg['hidden_size_layer_1']
    hidden_size_layer_2 = cfg['hidden_size_layer_2']
    hidden_size_rec = cfg['hidden_size_rec']
    hidden_size_pred = cfg['hidden_size_pred']
    dropout_encoder = cfg['dropout_encoder']
    dropout_rec = cfg['dropout_rec']
    dropout_pred = cfg['dropout_pred']
    softplus = cfg['softplus']

    logger.info('Loading model... ')

    model = RNN_VAE(TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES,FUTURE_DECODER,FUTURE_STEPS, hidden_size_layer_1,
                            hidden_size_layer_2, hidden_size_rec, hidden_size_pred, dropout_encoder,
                            dropout_rec, dropout_pred, softplus)
    if torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.cpu()

    model.load_state_dict(torch.load(os.path.join(cfg['project_path'],'model','best_model',model_name+'_'+cfg['Project']+'.pkl')))
    model.eval()

    return model

@save_state(model=GenerativeModelFunctionSchema)
def generative_model(config: str, mode: str = "sampling", save_logs: bool = False) -> plt.Figure:
    """Generative model.

    Args:
        config (str): Path to the configuration file.
        mode (str, optional): Mode for generating samples. Defaults to "sampling".

    Returns:
        plt.Figure: Plot of generated samples.
    """
    try:
        config_file = Path(config).resolve()
        cfg = read_config(config_file)
        if save_logs:
            logs_path = Path(cfg['project_path']) / "logs" / 'generative_model.log'
            logger_config.add_file_handler(logs_path)
        logger.info(f'Running generative model with mode {mode}...')
        model_name = cfg['model_name']
        n_cluster = cfg['n_cluster']
        parametrization = cfg['parametrization']

        files = []
        if cfg['all_data'] == 'No':
            all_flag = input("Do you want to write motif videos for your entire dataset? \n"
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


        model = load_model(cfg, model_name)

        for file in files:
            path_to_file=os.path.join(cfg['project_path'],"results",file,model_name, parametrization + '-' +str(n_cluster),"")

            if mode == "sampling":
                latent_vector = np.load(os.path.join(path_to_file,'latent_vector_'+file+'.npy'))
                return random_generative_samples(cfg, model, latent_vector)

            if mode == "reconstruction":
                latent_vector = np.load(os.path.join(path_to_file,'latent_vector_'+file+'.npy'))
                return random_reconstruction_samples(cfg, model, latent_vector)

            if mode == "centers":
                cluster_center = np.load(os.path.join(path_to_file,'cluster_center_'+file+'.npy'))
                return visualize_cluster_center(cfg, model, cluster_center)

            if mode == "motifs":
                latent_vector = np.load(os.path.join(path_to_file,'latent_vector_'+file+'.npy'))
                labels = np.load(os.path.join(path_to_file,"",str(n_cluster)+'_' + parametrization + '_label_'+file+'.npy'))
                return random_generative_samples_motif(cfg, model, latent_vector,labels,n_cluster)
    except Exception as e:
        logger.exception(str(e))
        raise
    finally:
        logger_config.remove_file_handler()