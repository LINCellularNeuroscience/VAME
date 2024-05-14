# -*- coding: utf-8 -*-
"""
Variational Animal Motion Embedding 0.1 Toolbox
© K. Luxem & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0

The Model is partially adapted from the Timeseries Clustering repository developed by Tejas Lodaya:
https://github.com/tejaslodaya/timeseries-clustering-vae/blob/master/vrae/vrae.py
"""


import torch
from torch import nn
from torch.autograd import Variable
from typing import Tuple

# NEW MODEL WITH SMALL ALTERATIONS
""" MODEL  """

class Encoder(nn.Module):
    """Encoder module of the Variational Autoencoder."""
    def __init__(self, NUM_FEATURES: int, hidden_size_layer_1: int, hidden_size_layer_2: int, dropout_encoder: float):
        """
        Initialize the Encoder module.

        Args:
            NUM_FEATURES (int): Number of input features.
            hidden_size_layer_1 (int): Size of the first hidden layer.
            hidden_size_layer_2 (int): Size of the second hidden layer.
            dropout_encoder (float): Dropout rate for regularization.
        """
        super(Encoder, self).__init__()

        self.input_size = NUM_FEATURES
        self.hidden_size = hidden_size_layer_1
        self.hidden_size_2 = hidden_size_layer_2
        self.n_layers  = 2
        self.dropout   = dropout_encoder
        self.bidirectional = True

        self.encoder_rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.n_layers,
                            bias=True, batch_first=True, dropout=self.dropout, bidirectional=self.bidirectional)#UNRELEASED!


        self.hidden_factor = (2 if self.bidirectional else 1) * self.n_layers

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Encoder module.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, sequence_length, num_features).

        Returns:
            torch.Tensor: Encoded representation tensor of shape (batch_size, hidden_size_layer_1 * 4).
        """
        outputs_1, hidden_1 = self.encoder_rnn(inputs)#UNRELEASED!

        hidden = torch.cat((hidden_1[0,...], hidden_1[1,...], hidden_1[2,...], hidden_1[3,...]),1)

        return hidden


class Lambda(nn.Module):
    """Lambda module for computing the latent space parameters."""
    def __init__(self, ZDIMS: int, hidden_size_layer_1: int, hidden_size_layer_2: int, softplus: bool):
        """
        Initialize the Lambda module.

        Args:
            ZDIMS (int): Size of the latent space.
            hidden_size_layer_1 (int): Size of the first hidden layer.
            hidden_size_layer_2 (int, deprecated): Size of the second hidden layer.
            softplus (bool): Whether to use softplus activation for logvar.
        """
        super(Lambda, self).__init__()

        self.hid_dim = hidden_size_layer_1*4
        self.latent_length = ZDIMS
        self.softplus = softplus

        self.hidden_to_mean = nn.Linear(self.hid_dim, self.latent_length)
        self.hidden_to_logvar = nn.Linear(self.hid_dim, self.latent_length)

        if self.softplus == True:
            print("Using a softplus activation to ensures that the variance is parameterized as non-negative and activated by a smooth function")
            self.softplus_fn = nn.Softplus()

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Lambda module.

        Args:
            hidden (torch.Tensor): Hidden representation tensor of shape (batch_size, hidden_size_layer_1 * 4).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Latent space tensor, mean tensor, logvar tensor.
        """
        self.mean = self.hidden_to_mean(hidden)
        if self.softplus == True:
            self.logvar = self.softplus_fn(self.hidden_to_logvar(hidden))
        else:
            self.logvar = self.hidden_to_logvar(hidden)

        if self.training:
            std = torch.exp(0.5 * self.logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(self.mean), self.mean, self.logvar
        else:
            return self.mean, self.mean, self.logvar


class Decoder(nn.Module):
    """Decoder module of the Variational Autoencoder."""
    def __init__(self, TEMPORAL_WINDOW: int, ZDIMS: int, NUM_FEATURES: int, hidden_size_rec: int, dropout_rec: float):
        """
        Initialize the Decoder module.

        Args:
            TEMPORAL_WINDOW (int): Size of the temporal window.
            ZDIMS (int): Size of the latent space.
            NUM_FEATURES (int): Number of input features.
            hidden_size_rec (int): Size of the recurrent hidden layer.
            dropout_rec (float): Dropout rate for regularization.
        """
        super(Decoder,self).__init__()

        self.num_features = NUM_FEATURES
        self.sequence_length = TEMPORAL_WINDOW
        self.hidden_size = hidden_size_rec
        self.latent_length = ZDIMS
        self.n_layers  = 1
        self.dropout   = dropout_rec
        self.bidirectional = True

        self.rnn_rec = nn.GRU(self.latent_length, hidden_size=self.hidden_size, num_layers=self.n_layers,
                            bias=True, batch_first=True, dropout=self.dropout, bidirectional=self.bidirectional)

        self.hidden_factor = (2 if self.bidirectional else 1) * self.n_layers # NEW

        self.latent_to_hidden = nn.Linear(self.latent_length,self.hidden_size * self.hidden_factor) # NEW
        self.hidden_to_output = nn.Linear(self.hidden_size*(2 if self.bidirectional else 1), self.num_features)

    def forward(self, inputs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Decoder module.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, seq_len, ZDIMS).
            z (torch.Tensor): Latent space tensor of shape (batch_size, ZDIMS).

        Returns:
            torch.Tensor: Decoded output tensor of shape (batch_size, seq_len, NUM_FEATURES).
        """
        batch_size = inputs.size(0) # NEW

        hidden = self.latent_to_hidden(z) # NEW

        hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size) # NEW

        decoder_output, _ = self.rnn_rec(inputs, hidden)
        prediction = self.hidden_to_output(decoder_output)

        return prediction


class Decoder_Future(nn.Module):
    """Decoder module for predicting future sequences."""
    def __init__(
        self,
        TEMPORAL_WINDOW: int,
        ZDIMS: int,
        NUM_FEATURES: int,
        FUTURE_STEPS: int,
        hidden_size_pred: int,
        dropout_pred: float
    ):
        """
        Initialize the Decoder_Future module.

        Args:
            TEMPORAL_WINDOW (int): Size of the temporal window.
            ZDIMS (int): Size of the latent space.
            NUM_FEATURES (int): Number of input features.
            FUTURE_STEPS (int): Number of future steps to predict.
            hidden_size_pred (int): Size of the prediction hidden layer.
            dropout_pred (float): Dropout rate for regularization.
        """
        super(Decoder_Future,self).__init__()

        self.num_features = NUM_FEATURES
        self.future_steps = FUTURE_STEPS
        self.sequence_length = TEMPORAL_WINDOW
        self.hidden_size = hidden_size_pred
        self.latent_length = ZDIMS
        self.n_layers  = 1
        self.dropout   = dropout_pred
        self.bidirectional = True

        self.rnn_pred = nn.GRU(self.latent_length, hidden_size=self.hidden_size, num_layers=self.n_layers,
                            bias=True, batch_first=True, dropout=self.dropout, bidirectional=self.bidirectional)

        self.hidden_factor = (2 if self.bidirectional else 1) * self.n_layers # NEW

        self.latent_to_hidden = nn.Linear(self.latent_length,self.hidden_size * self.hidden_factor)
        self.hidden_to_output = nn.Linear(self.hidden_size*2, self.num_features)

    def forward(self, inputs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Decoder_Future module.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, seq_len, ZDIMS).
            z (torch.Tensor): Latent space tensor of shape (batch_size, ZDIMS).

        Returns:
            torch.Tensor: Predicted future tensor of shape (batch_size, FUTURE_STEPS, NUM_FEATURES).
        """
        batch_size = inputs.size(0)

        hidden = self.latent_to_hidden(z)
        hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)

        inputs = inputs[:,:self.future_steps,:]
        decoder_output, _ = self.rnn_pred(inputs, hidden)

        prediction = self.hidden_to_output(decoder_output)

        return prediction


class RNN_VAE(nn.Module):
    """Variational Autoencoder module."""
    def __init__(
        self,
        TEMPORAL_WINDOW: int,
        ZDIMS: int,
        NUM_FEATURES: int,
        FUTURE_DECODER: bool,
        FUTURE_STEPS: int,
        hidden_size_layer_1: int,
        hidden_size_layer_2: int,
        hidden_size_rec: int,
        hidden_size_pred: int,
        dropout_encoder: float,
        dropout_rec: float,
        dropout_pred: float,
        softplus: bool
    ):

        """
        Initialize the VAE module.

        Args:
            TEMPORAL_WINDOW (int): Size of the temporal window.
            ZDIMS (int): Size of the latent space.
            NUM_FEATURES (int): Number of input features.
            FUTURE_DECODER (bool): Whether to include a future decoder.
            FUTURE_STEPS (int): Number of future steps to predict.
            hidden_size_layer_1 (int): Size of the first hidden layer.
            hidden_size_layer_2 (int): Size of the second hidden layer.
            hidden_size_rec (int): Size of the recurrent hidden layer.
            hidden_size_pred (int): Size of the prediction hidden layer.
            dropout_encoder (float): Dropout rate for encoder.

        """
        super(RNN_VAE,self).__init__()

        self.FUTURE_DECODER = FUTURE_DECODER
        self.seq_len = int(TEMPORAL_WINDOW / 2)
        self.encoder = Encoder(NUM_FEATURES, hidden_size_layer_1, hidden_size_layer_2, dropout_encoder)
        self.lmbda = Lambda(ZDIMS, hidden_size_layer_1, hidden_size_layer_2, softplus)
        self.decoder = Decoder(self.seq_len,ZDIMS,NUM_FEATURES, hidden_size_rec, dropout_rec)
        if FUTURE_DECODER:
            self.decoder_future = Decoder_Future(self.seq_len,ZDIMS,NUM_FEATURES,FUTURE_STEPS, hidden_size_pred,
                                                 dropout_pred)

    def forward(self, seq: torch.Tensor) -> tuple:
        """Forward pass of the VAE.

        Args:
            seq (torch.Tensor): Input sequence tensor of shape (batch_size, seq_len, NUM_FEATURES).

        Returns:
            Tuple containing:
                - If FUTURE_DECODER is True:
                    - prediction (torch.Tensor): Reconstructed input sequence tensor.
                    - future (torch.Tensor): Predicted future sequence tensor.
                    - z (torch.Tensor): Latent representation tensor.
                    - mu (torch.Tensor): Mean of the latent distribution tensor.
                    - logvar (torch.Tensor): Log variance of the latent distribution tensor.
                - If FUTURE_DECODER is False:
                    - prediction (torch.Tensor): Reconstructed input sequence tensor.
                    - z (torch.Tensor): Latent representation tensor.
                    - mu (torch.Tensor): Mean of the latent distribution tensor.
                    - logvar (torch.Tensor): Log variance of the latent distribution tensor.
        """

        """ Encode input sequence """
        h_n = self.encoder(seq)

        """ Compute the latent state via reparametrization trick """
        z, mu, logvar = self.lmbda(h_n)
        ins = z.unsqueeze(2).repeat(1, 1, self.seq_len)
        ins = ins.permute(0,2,1)

        """ Predict the future of the sequence from the latent state"""
        prediction = self.decoder(ins, z)

        if self.FUTURE_DECODER:
            future = self.decoder_future(ins, z)
            return prediction, future, z, mu, logvar
        else:
            return prediction, z, mu, logvar


#----------------------------------------------------------------------------------------
#                               LEGACY MODEL                                            |
#----------------------------------------------------------------------------------------


""" MODEL """
class Encoder_LEGACY(nn.Module):
    """LEGACY Encoder module of the Variational Autoencoder."""
    def __init__(self, NUM_FEATURES: int, hidden_size_layer_1: int, hidden_size_layer_2: int, dropout_encoder: float):
        """(LEGACY) Initialize the Encoder_LEGACY module.

        Args:
            NUM_FEATURES (int): Number of input features.
            hidden_size_layer_1 (int): Size of the first hidden layer.
            hidden_size_layer_2 (int): Size of the second hidden layer.
            dropout_encoder (float): Dropout rate for the encoder.
        """
        super(Encoder_LEGACY, self).__init__()

        self.input_size = NUM_FEATURES
        self.hidden_size = hidden_size_layer_1
        self.hidden_size_2 = hidden_size_layer_2
        self.n_layers  = 1
        self.dropout   = dropout_encoder

        self.rnn_1 = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.n_layers,
                            bias=True, batch_first=True, dropout=self.dropout, bidirectional=True)

        self.rnn_2 = nn.GRU(input_size=self.hidden_size*2, hidden_size=self.hidden_size_2, num_layers=self.n_layers,
                            bias=True, batch_first=True, dropout=self.dropout, bidirectional=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """(LEGACY) Forward pass of the Encoder_LEGACY module.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, seq_len, NUM_FEATURES).

        Returns:
            torch.Tensor: Encoded tensor.
        """
        outputs_1, hidden_1 = self.rnn_1(inputs)
        outputs_2, hidden_2 = self.rnn_2(outputs_1)

        h_n_1 = torch.cat((hidden_1[0,...], hidden_1[1,...]), 1)
        h_n_2 = torch.cat((hidden_2[0,...], hidden_2[1,...]), 1)

        h_n = torch.cat((h_n_1, h_n_2), 1)

        return h_n


class Lambda_LEGACY(nn.Module):
    """LEGACY Lambda module for computing the latent space parameters."""
    def __init__(self, ZDIMS: int, hidden_size_layer_1: int, hidden_size_layer_2: int):
        """(LEGACY) Initialize the Lambda_LEGACY module.

        Args:
            ZDIMS (int): Size of the latent space.
            hidden_size_layer_1 (int): Size of the first hidden layer.
            hidden_size_layer_2 (int): Size of the second hidden layer.
        """
        super(Lambda_LEGACY, self).__init__()

        self.hid_dim = hidden_size_layer_1*2 + hidden_size_layer_2*2
        self.latent_length = ZDIMS

        self.hidden_to_linear = nn.Linear(self.hid_dim, self.hid_dim)
        self.hidden_to_mean = nn.Linear(self.hid_dim, self.latent_length)
        self.hidden_to_logvar = nn.Linear(self.hid_dim, self.latent_length)

        self.softplus = nn.Softplus()

    def forward(self, cell_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(LEGACY) Forward pass of the Lambda_LEGACY module.

        Args:
            cell_output (torch.Tensor): Output tensor of the encoder.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing:
                - torch.Tensor: Sampled latent tensor.
                - torch.Tensor: Mean of the latent distribution.
                - torch.Tensor: Log variance of the latent distribution.
        """
        self.latent_mean = self.hidden_to_mean(cell_output)

        # based on Pereira et al 2019:
        # "The SoftPlus function ensures that the variance is parameterized as non-negative and activated
        # by a smooth function
        self.latent_logvar = self.softplus(self.hidden_to_logvar(cell_output))

        if self.training:
            std = self.latent_logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(self.latent_mean), self.latent_mean, self.latent_logvar
        else:
            return self.latent_mean, self.latent_mean, self.latent_logvar


class Decoder_LEGACY(nn.Module):
    """LEGACY Decoder module of the Variational Autoencoder."""
    def __init__(self, TEMPORAL_WINDOW: int, ZDIMS: int, NUM_FEATURES: int, hidden_size_rec: int, dropout_rec: float):
        """(LEGACY) Initialize the Decoder_LEGACY module.

        Args:
            TEMPORAL_WINDOW (int): Size of the temporal window.
            ZDIMS (int): Size of the latent space.
            NUM_FEATURES (int): Number of input features.
            hidden_size_rec (int): Size of the recurrent hidden layer.
            dropout_rec (float): Dropout rate for the decoder.
        """
        super(Decoder_LEGACY,self).__init__()

        self.num_features = NUM_FEATURES
        self.sequence_length = TEMPORAL_WINDOW
        self.hidden_size = hidden_size_rec
        self.latent_length = ZDIMS
        self.n_layers  = 1
        self.dropout   = dropout_rec

        self.rnn_rec = nn.GRU(self.latent_length, hidden_size=self.hidden_size, num_layers=self.n_layers,
                            bias=True, batch_first=True, dropout=self.dropout, bidirectional=False)

        self.hidden_to_output = nn.Linear(self.hidden_size, self.num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """(LEGACY) Forward pass of the Decoder_LEGACY module.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Reconstructed tensor.
        """
        decoder_output, _ = self.rnn_rec(inputs)
        prediction = self.hidden_to_output(decoder_output)

        return prediction

class Decoder_Future_LEGACY(nn.Module):
    """LEGACY Decoder module for predicting future sequences."""
    def __init__(self, TEMPORAL_WINDOW: int, ZDIMS: int, NUM_FEATURES: int, FUTURE_STEPS: int, hidden_size_pred: int, dropout_pred: float):
        """(LEGACY) Initialize the Decoder_Future_LEGACY module.

        Args:
            TEMPORAL_WINDOW (int): Size of the temporal window.
            ZDIMS (int): Size of the latent space.
            NUM_FEATURES (int): Number of input features.
            FUTURE_STEPS (int): Number of future steps to predict.
            hidden_size_pred (int): Size of the prediction hidden layer.
            dropout_pred (float): Dropout rate for the prediction.
        """
        super(Decoder_Future_LEGACY,self).__init__()

        self.num_features = NUM_FEATURES
        self.future_steps = FUTURE_STEPS
        self.sequence_length = TEMPORAL_WINDOW
        self.hidden_size = hidden_size_pred
        self.latent_length = ZDIMS
        self.n_layers  = 1
        self.dropout   = dropout_pred

        self.rnn_pred = nn.GRU(self.latent_length, hidden_size=self.hidden_size, num_layers=self.n_layers,
                            bias=True, batch_first=True, dropout=self.dropout, bidirectional=True)

        self.hidden_to_output = nn.Linear(self.hidden_size*2, self.num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """(LEGACY) Forward pass of the Decoder_Future_LEGACY module.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Predicted future tensor.
        """
        inputs = inputs[:,:self.future_steps,:]
        decoder_output, _ = self.rnn_pred(inputs)
        prediction = self.hidden_to_output(decoder_output)

        return prediction


class RNN_VAE_LEGACY(nn.Module):
    """LEGACY Variational Autoencoder module."""
    def __init__(
        self,
        TEMPORAL_WINDOW: int,
        ZDIMS: int,
        NUM_FEATURES: int,
        FUTURE_DECODER: bool,
        FUTURE_STEPS: int,
        hidden_size_layer_1: int,
        hidden_size_layer_2: int,
        hidden_size_rec: int,
        hidden_size_pred: int,
        dropout_encoder: float,
        dropout_rec: float,
        dropout_pred: float,
        softplus: bool
    ):
        """(LEGACY) Initialize the RNN_VAE_LEGACY module.

        Args:
            TEMPORAL_WINDOW (int): Size of the temporal window.
            ZDIMS (int): Size of the latent space.
            NUM_FEATURES (int): Number of input features.
            FUTURE_DECODER (bool): Whether to include a future decoder.
            FUTURE_STEPS (int): Number of future steps to predict.
            hidden_size_layer_1 (int): Size of the first hidden layer.
            hidden_size_layer_2 (int): Size of the second hidden layer.
            hidden_size_rec (int): Size of the recurrent hidden layer.
            hidden_size_pred (int): Size of the prediction hidden layer.
            dropout_encoder (float): Dropout rate for the encoder.
            dropout_rec (float): Dropout rate for the decoder.
            dropout_pred (float): Dropout rate for the prediction.
            softplus (bool, deprecated): Whether to use softplus activation.
        """
        super(RNN_VAE_LEGACY,self).__init__()

        self.FUTURE_DECODER = FUTURE_DECODER
        self.seq_len = int(TEMPORAL_WINDOW / 2)
        self.encoder = Encoder_LEGACY(NUM_FEATURES, hidden_size_layer_1, hidden_size_layer_2, dropout_encoder)
        self.lmbda = Lambda_LEGACY(ZDIMS, hidden_size_layer_1, hidden_size_layer_2)
        self.decoder = Decoder_LEGACY(self.seq_len,ZDIMS,NUM_FEATURES, hidden_size_rec, dropout_rec)
        if FUTURE_DECODER:
            self.decoder_future = Decoder_Future_LEGACY(self.seq_len,ZDIMS,NUM_FEATURES,FUTURE_STEPS, hidden_size_pred,
                                                 dropout_pred)

    def forward(self, seq: torch.Tensor) -> Tuple:
        """Forward pass of the RNN_VAE_LEGACY module.

        Args:
            seq (torch.Tensor): Input sequence tensor of shape (batch_size, seq_len, NUM_FEATURES).

        Returns:
            Tuple: Tuple containing:
                - torch.Tensor: Predicted tensor.
                - torch.Tensor: Future prediction tensor if FUTURE_DECODER is True, else nothing.
                - torch.Tensor: Latent tensor.
                - torch.Tensor: Mean of the latent distribution.
                - torch.Tensor: Log variance of the latent distribution.
        """

        """ Encode input sequence """
        h_n = self.encoder(seq)

        """ Compute the latent state via reparametrization trick """
        latent, mu, logvar = self.lmbda(h_n)
        z = latent.unsqueeze(2).repeat(1, 1, self.seq_len)
        z = z.permute(0,2,1)

        """ Predict the future of the sequence from the latent state"""
        prediction = self.decoder(z)

        if self.FUTURE_DECODER:
            future = self.decoder_future(z)
            return prediction, future, latent, mu, logvar
        else:
            return prediction, latent, mu, logvar

