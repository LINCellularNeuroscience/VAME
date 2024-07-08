
import os
import yaml
import ruamel.yaml
from pathlib import Path
from typing import Tuple
import torch
from vame.logging.logger import VameLogger
from vame.model.rnn_model import RNN_VAE


logger_config = VameLogger(__name__)
logger = logger_config.logger

def load_model(cfg: dict, model_name: str, fixed: bool = True) -> RNN_VAE:
    """Load the VAME model.

    Args:
        cfg (dict): Configuration dictionary.
        model_name (str): Name of the model.
        fixed (bool): Fixed or variable length sequences.

    Returns:
        RNN_VAE: Loaded VAME model.
    """
    # load Model
    ZDIMS = cfg['zdims']
    FUTURE_DECODER = cfg['prediction_decoder']
    TEMPORAL_WINDOW = cfg['time_window']*2
    FUTURE_STEPS = cfg['prediction_steps']
    NUM_FEATURES = cfg['num_features']

    if not fixed:
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

    model = RNN_VAE(
        TEMPORAL_WINDOW,
        ZDIMS,
        NUM_FEATURES,
        FUTURE_DECODER,
        FUTURE_STEPS,
        hidden_size_layer_1,
        hidden_size_layer_2,
        hidden_size_rec,
        hidden_size_pred,
        dropout_encoder,
        dropout_rec,
        dropout_pred,
        softplus
    )
    if torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.cpu()

    model.load_state_dict(torch.load(os.path.join(cfg['project_path'],'model','best_model',model_name+'_'+cfg['Project']+'.pkl')))
    model.eval()

    return model
