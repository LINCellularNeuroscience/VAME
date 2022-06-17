import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
from torchviz import make_dot
import os
import numpy as np
from pathlib import Path
import torch
from torch import nn
from vame.util.auxiliary import read_config
from vame.model.dataloader import SEQUENCE_DATASET
from vame.model.rnn_model import RNN_VAE
import os
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'

class Encoder(nn.Module):
    def __init__(self, NUM_FEATURES, hidden_size_layer_1, hidden_size_layer_2, dropout_encoder):
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
        
    def forward(self, inputs):        
        outputs_1, hidden_1 = self.encoder_rnn(inputs)#UNRELEASED!
        
        hidden = torch.cat((hidden_1[0,...], hidden_1[1,...], hidden_1[2,...], hidden_1[3,...]),1)
        
        return hidden
    
    
class Lambda(nn.Module):
    def __init__(self,ZDIMS, hidden_size_layer_1, hidden_size_layer_2, softplus):
        super(Lambda, self).__init__()
        
        self.hid_dim = hidden_size_layer_1*4
        self.latent_length = ZDIMS
        self.softplus = softplus
        
        self.hidden_to_mean = nn.Linear(self.hid_dim, self.latent_length)
        self.hidden_to_logvar = nn.Linear(self.hid_dim, self.latent_length)
        
        if self.softplus == True:
            print("Using a softplus activation to ensures that the variance is parameterized as non-negative and activated by a smooth function")
            self.softplus_fn = nn.Softplus()
        
    def forward(self, hidden):
        
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
    def __init__(self,TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES, hidden_size_rec, dropout_rec):
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
        
    def forward(self, inputs, z):
        batch_size = inputs.size(0) # NEW
        
        hidden = self.latent_to_hidden(z) # NEW
        
        hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size) # NEW
        
        decoder_output, _ = self.rnn_rec(inputs, hidden)
        prediction = self.hidden_to_output(decoder_output)
        
        return prediction
    
    
class Decoder_Future(nn.Module):
    def __init__(self,TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES,FUTURE_STEPS, hidden_size_pred, dropout_pred):
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
        
    def forward(self, inputs, z):
        batch_size = inputs.size(0)
        
        hidden = self.latent_to_hidden(z)
        hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        
        inputs = inputs[:,:self.future_steps,:]
        decoder_output, _ = self.rnn_pred(inputs, hidden)
        
        prediction = self.hidden_to_output(decoder_output)
         
        return prediction


class RNN_VAE(nn.Module):
    def __init__(self,TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES,FUTURE_DECODER,FUTURE_STEPS, hidden_size_layer_1, 
                        hidden_size_layer_2, hidden_size_rec, hidden_size_pred, dropout_encoder, 
                        dropout_rec, dropout_pred, softplus):
        super(RNN_VAE,self).__init__()
        
        self.FUTURE_DECODER = FUTURE_DECODER
        self.seq_len = int(TEMPORAL_WINDOW / 2)
        self.encoder = Encoder(NUM_FEATURES, hidden_size_layer_1, hidden_size_layer_2, dropout_encoder)
        self.lmbda = Lambda(ZDIMS, hidden_size_layer_1, hidden_size_layer_2, softplus)
        self.decoder = Decoder(self.seq_len,ZDIMS,NUM_FEATURES, hidden_size_rec, dropout_rec)
        if FUTURE_DECODER:
            self.decoder_future = Decoder_Future(self.seq_len,ZDIMS,NUM_FEATURES,FUTURE_STEPS, hidden_size_pred,
                                                 dropout_pred)
        
    def forward(self,seq):
        
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


def modeltorchviz(model,input2):
    # from torchviz import make_dot
    # params: model = MSDNet(args).cuda()
    # params: input = (3, 32, 32) 
    # params: input2 = torch.randn(1, 1, 28, 28).requires_grad_(True)  # 定义一个网络的输入值
    print(model)        
    # summary(model, input)
    y = model(input2.cuda())    # 获取网络的预测值

    MyConvNetVis = make_dot(y, params=dict(list(model.named_parameters()) + [('x', input2)]))
    MyConvNetVis.format = "png"
    # 指定文件生成的文件夹
    MyConvNetVis.directory = "data"
    # 生成文件
    MyConvNetVis.view() 
    
def vis_rnn_network(config):  
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    TEMPORAL_WINDOW = cfg['time_window']*2
    ZDIMS = cfg['zdims']
    NUM_FEATURES = cfg['num_features']
    FUTURE_DECODER = cfg['prediction_decoder']
    FUTURE_STEPS = cfg['prediction_steps']
    hidden_size_layer_1 = cfg['hidden_size_layer_1']
    hidden_size_layer_2 = cfg['hidden_size_layer_2']
    hidden_size_rec = cfg['hidden_size_rec']
    hidden_size_pred = cfg['hidden_size_pred']
    dropout_encoder = cfg['dropout_encoder']
    dropout_rec = cfg['dropout_rec']
    dropout_pred = cfg['dropout_pred']
    softplus = cfg['softplus']

    model = RNN_VAE(TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES,FUTURE_DECODER,FUTURE_STEPS,hidden_size_layer_1,
    hidden_size_layer_2,hidden_size_rec,hidden_size_pred,dropout_encoder,dropout_rec,dropout_pred,softplus).cuda()
    input2 = torch.randn(1,1,30).requires_grad_(True)  # 定义一个网络的输入值
    modeltorchviz(model,input2)
    