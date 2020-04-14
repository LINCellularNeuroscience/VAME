#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 10:46:12 2019

@author: luxemk
"""

import torch
from torch import nn
import torch.utils.data as Data
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

import os
import numpy as np
from pathlib import Path

from VAME.util.auxiliary import read_config
from VAME.model.dataloader.temporal_dataloader import SEQUENCE_DATASET
from VAME.model.dataloader.spatial_features_dataloader import SPATIAL_FEATURES

""" MODEL """
class Encoder(nn.Module):
    def __init__(self, NUM_FEATURES):
        super(Encoder, self).__init__()
        
        self.input_size = NUM_FEATURES
        self.hidden_size = 256
        self.hidden_size_2 = 256
        self.n_layers  = 1
        self.dropout   = 0.2
        
#        self.conv_layer = nn.Sequential(
#                nn.Conv1d(12, 12, 1),
##                nn.Dropout(p=0.2),
#                nn.BatchNorm1d(12),
#                nn.LeakyReLU(),
#                nn.MaxPool1d(2,2)
#                )
        
        self.lstm_1 = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.n_layers,
                            bias=True, batch_first=True, dropout=self.dropout, bidirectional=True)
        
        self.lstm_2 = nn.GRU(input_size=self.hidden_size*2, hidden_size=self.hidden_size_2, num_layers=self.n_layers,
                            bias=True, batch_first=True, dropout=self.dropout, bidirectional=True)
        
    def forward(self, inputs):
#        inputs = inputs.permute(0,2,1)
#        conv_out = self.conv_layer(inputs)
#        conv_out = conv_out.permute(0,2,1)
        outputs_1, hidden_1 = self.lstm_1(inputs)
        outputs_2, hidden_2 = self.lstm_2(outputs_1)
        
        h_n_1 = torch.cat((hidden_1[0,...], hidden_1[1,...]), 1)
        h_n_2 = torch.cat((hidden_2[0,...], hidden_2[1,...]), 1)

        h_n = torch.cat((h_n_1, h_n_2), 1)
        
        return h_n
    
    
class Lambda(nn.Module):
    """Lambda module converts output of encoder to latent vector

    :param hidden_size: hidden size of the encoder
    :param latent_length: latent vector length
    """
    def __init__(self,ZDIMS):
        super(Lambda, self).__init__()

        self.hid_dim = 1024
        self.latent_length = ZDIMS
        
        self.hidden_to_linear = nn.Linear(self.hid_dim, self.hid_dim)
        self.hidden_to_mean = nn.Linear(self.hid_dim, self.latent_length)
        self.hidden_to_logvar = nn.Linear(self.hid_dim, self.latent_length)

#        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
#        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)
        
        self.softplus = nn.Softplus()

    def forward(self, cell_output):
        """Given last hidden state of encoder, passes through a linear layer, and finds the mean and variance

        :param cell_output: last hidden state of encoder
        :return: latent vector
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
  
      
class Decoder(nn.Module):
    def __init__(self,TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES):
        super(Decoder,self).__init__()
        
        self.num_features = NUM_FEATURES
        self.sequence_length = TEMPORAL_WINDOW
        self.batch_size = 64
        self.hidden_size = 256
        self.latent_length = ZDIMS
        self.n_layers  = 1
        self.dropout   = 0.2
        
        self.gru = nn.GRU(self.latent_length, hidden_size=self.hidden_size, num_layers=self.n_layers,
                            bias=True, batch_first=True, dropout=self.dropout, bidirectional=False)

        self.hidden_to_output = nn.Linear(self.hidden_size, self.num_features)

    def forward(self, inputs):
        decoder_output, _ = self.gru(inputs)
        prediction = self.hidden_to_output(decoder_output)
        
        return prediction
    
class Decoder_Future(nn.Module):
    def __init__(self,TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES,FUTURE_STEPS):
        super(Decoder_Future,self).__init__()
        
        self.num_features = NUM_FEATURES
        self.future_steps = FUTURE_STEPS
        self.sequence_length = TEMPORAL_WINDOW
        self.batch_size = 64
        self.hidden_size = 256
        self.latent_length = ZDIMS
        self.n_layers  = 1
        self.dropout   = 0.2
        
        self.gru = nn.GRU(self.latent_length, hidden_size=self.hidden_size, num_layers=self.n_layers,
                            bias=True, batch_first=True, dropout=self.dropout, bidirectional=True)
        
        self.hidden_to_output = nn.Linear(self.hidden_size*2, self.num_features)
        
    def forward(self, inputs):
        inputs = inputs[:,:15,:]
        decoder_output, _ = self.gru(inputs)
        prediction = self.hidden_to_output(decoder_output)
        
        return prediction


class RNN_VAE(nn.Module):
    def __init__(self,TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES,FUTURE_DECODER,FUTURE_STEPS):
        super(RNN_VAE,self).__init__()
        
        self.n_cluster = 25
        self.batch_size = 64
        self.FUTURE_DECODER = FUTURE_DECODER
        self.seq_len = int(TEMPORAL_WINDOW / 2)
        self.encoder = Encoder(NUM_FEATURES)
        self.lmbda = Lambda(ZDIMS)
        self.decoder = Decoder(self.seq_len,ZDIMS,NUM_FEATURES)
        if FUTURE_DECODER:
            self.decoder_future = Decoder_Future(self.seq_len,ZDIMS,NUM_FEATURES,FUTURE_STEPS)
        
    def forward(self,seq):
        
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
            return prediction, latent, mu, logvar#, self.F


def reconstruction_loss(x, x_tilde):
    batch_size = x.shape[0]
    mse_loss = nn.MSELoss(reduction='sum')
#    mse_loss = nn.SmoothL1Loss(reduction='sum') # 'mean'
    rec_loss = mse_loss(x_tilde,x)
    return rec_loss #/ batch_size

def future_reconstruction_loss(x, x_tilde):
    batch_size = x.shape[0]
    mse_loss = nn.MSELoss(reduction='sum')
#    mse_loss = nn.SmoothL1Loss(reduction='sum') # 'mean'
    rec_loss = mse_loss(x_tilde,x)
    return rec_loss #/ batch_size

def cluster_loss(H, lmbda=1):
    batch_size = H.shape[0]
    gram_matrix = H.T @ H 
    _ ,sv_2, _ = torch.svd(gram_matrix)
    sv = torch.sqrt(sv_2[:30])
    loss = torch.sum(sv)
    return lmbda*loss / batch_size
    

def kullback_leibler_loss(mu, logvar):
    # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # I'm using torch.mean() here as the sum() version depends on the size of the latent vector
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
#    kl_divergence = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
#    KLD = kl_divergence.sum(1).mean(0, True)
    return KLD


def kl_annealing(epoch, kl_start, annealtime, function='linear'):
    """
        Annealing of Kullback-Leibler loss to let the model learn first
        the reconstruction of the data before the KL loss term gets introduced.
    """
    if epoch > kl_start:
        if function == 'linear':
            new_weight = min(1, epoch/annealtime)
            
        elif function == 'sigmoid':
            new_weight = float(1/(1+np.exp(-0.9*(epoch-annealtime))))
        else:
            raise NotImplementedError('currently only "linear" and "sigmoid" are implemented')
        
        return new_weight
    
    else:
        new_weight = 0
        return new_weight
    

def gaussian(ins, is_training, seq_len, std_n=0.8):
    if is_training:
        emp_std = ins.std(1)*std_n
        emp_std = emp_std.unsqueeze(2).repeat(1, 1, seq_len)
        emp_std = emp_std.permute(0,2,1)
        noise = Variable(ins.data.new(ins.size()).normal_(0, 1))
        return ins + (noise*emp_std)
    return ins


def train(train_loader, epoch, model, optimizer, anneal_function, BETA, kl_start, 
          annealtime, seq_len, future_decoder, future_steps, scheduler):
    model.train() # toggle model to train mode
    train_loss = 0.0
    mse_loss = 0.0
    kullback_loss = 0.0
    kmeans_losses = 0.0
    fut_loss = 0.0
    lmbda = 1
    
    seq_len_half = int(seq_len / 2)
    
    for idx, data_item in enumerate(train_loader):             
        data_item = Variable(data_item)
        data_item = data_item.permute(0,2,1)
        data = data_item[:,:seq_len_half,:].type('torch.FloatTensor').cuda()
        fut = data_item[:,seq_len_half:45,:].type('torch.FloatTensor').cuda()
        data_gaussian = gaussian(data,True, seq_len_half)
#        data_gaussian = data
                                 
        if future_decoder:
            data_tilde, future, latent, mu, logvar = model(data_gaussian)
        
            rec_loss = reconstruction_loss(data, data_tilde)
            fut_rec_loss = future_reconstruction_loss(fut, future)
            kmeans_loss = cluster_loss(latent.T)
#            kmeans_loss=0
            kl_loss = kullback_leibler_loss(mu, logvar)
            kl_weight = kl_annealing(epoch, kl_start, annealtime, anneal_function)
            loss = rec_loss + lmbda*fut_rec_loss + BETA*kl_weight*kl_loss + kl_weight*kmeans_loss
            
            fut_loss += lmbda*fut_rec_loss.item()
        else:
            data_tilde, latent, mu, logvar = model(data_gaussian)
        
            rec_loss = reconstruction_loss(data, data_tilde)
            kl_loss = kullback_leibler_loss(mu, logvar)
            kmeans_loss = cluster_loss(latent.T)
            kl_weight = kl_annealing(epoch, kl_start, annealtime, anneal_function)
            loss = rec_loss + BETA*kl_weight*kl_loss + kl_weight*kmeans_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 5)
        
        train_loss += loss.item()
        mse_loss += rec_loss.item()
        kullback_loss += kl_loss.item()
        kmeans_losses += kmeans_loss
        
        if idx % 1000 == 0:
            print('Epoch: %d.  loss: %.4f' %(epoch, loss.item()))
    
    scheduler.step()
    
    if future_decoder:    
        print('Average Train loss: {:.4f}, MSE-Loss: {:.4f}, MSE-Future-Loss {:.4f}, KL-Loss: {:.4f}, KL-weigt: {:.4f},  Kmeans-Loss: {:.4f}'.format(train_loss / idx,
              mse_loss /idx, fut_loss/idx, BETA*kl_weight*kullback_loss/idx, kl_weight, kl_weight*kmeans_losses/idx))
    else:
        print('Average Train loss: {:.4f}, MSE-Loss: {:.4f}, KL-Loss: {:.4f}, weight: {:.4f}, Kmeans-Loss: {:.4f}'.format(train_loss / idx,
              mse_loss /idx, BETA*kl_weight*kullback_loss/idx, kl_weight, kl_weight*kmeans_losses/idx))
    
    return kl_weight, train_loss/idx, kl_weight*kmeans_losses/idx, kullback_loss/idx, mse_loss/idx, fut_loss/idx


def test(test_loader, epoch, model, optimizer, BETA, kl_weight, seq_len, future_decoder, future_steps):
    model.eval() # toggle model to inference mode
    test_loss = 0.0
    mse_loss = 0.0
    fut_loss = 0.0
    kullback_loss = 0.0
    kmeans_losses = 0.0
    seq_len_half = int(seq_len / 2)
    
    lmbda = 1
    
    with torch.no_grad():
        for idx, data_item in enumerate(test_loader):
            # we're only going to infer, so no autograd at all required: volatile=True
            data_item = Variable(data_item)
            data_item = data_item.permute(0,2,1)
            data = data_item[:,:seq_len_half,:].type('torch.FloatTensor').cuda()
#            fut = data_item[:,seq_len_half:,:].type('torch.FloatTensor').cuda()
            
            if future_decoder:
                recon_images, future, latent, mu, logvar = model(data)
            
                rec_loss = reconstruction_loss(data, recon_images)
#                fut_rec_loss = future_reconstruction_loss(fut, future)
                fut_rec_loss = 0
                kl_loss = kullback_leibler_loss(mu, logvar)
                kmeans_loss = cluster_loss(latent.T)
                loss = rec_loss + lmbda*fut_rec_loss + BETA*kl_weight*kl_loss+ kl_weight*kmeans_loss
                
                fut_loss += lmbda*fut_rec_loss
                
            else:
                recon_images, latent, mu, logvar = model(data)
            
                rec_loss = reconstruction_loss(data, recon_images)
                kl_loss = kullback_leibler_loss(mu, logvar)
                kmeans_loss = cluster_loss(latent.T)
                loss = rec_loss + BETA*kl_weight*kl_loss + kl_weight*kmeans_loss
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 10)
            
            test_loss += loss.item()
            mse_loss += rec_loss.item()
            kullback_loss += kl_loss.item()
            kmeans_losses += kmeans_loss

    if future_decoder:    
        print('Average Test loss: {:.4f}, MSE-Loss: {:.4f}, MSE-Future-Loss {:.4f}, KL-Loss: {:.4f}, KL-weigt: {:.4f},  Kmeans-Loss: {:.4f}'.format(test_loss / idx,
              mse_loss /idx, fut_loss/idx, BETA*kl_weight*kullback_loss/idx, kl_weight, kl_weight*kmeans_losses/idx))
    else:
        print('Average Test loss: {:.4f}, MSE-Loss: {:.4f}, KL-Loss: {:.4f}, Kmeans-Loss: {:.4f}'.format(test_loss / idx,
              mse_loss /idx, BETA*kl_weight*kullback_loss/idx, kl_weight*kmeans_losses/idx))
    
    return mse_loss /idx, test_loss/idx, kl_weight*kmeans_losses


def temporal_lstm(config, model_name, spatial_features=False, pretrained=False, debug=False):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    
    FUTURE_DECODER = cfg['future_decoder']
    
    if FUTURE_DECODER:
        print("Train composite temporal model!")
        folder='temporal_model'
        if not os.path.exists(cfg['project_path']+'/'+'model/temporal_model/best_model'):
            os.mkdir(cfg['project_path']+'/model/temporal_model')
            os.mkdir(cfg['project_path']+'/model/'+'temporal_model/best_model')
            os.mkdir(cfg['project_path']+'/model/'+'temporal_model/losses')
    else:
        print("Train temporal model!")
        folder='temporal_model'
        if not os.path.exists(cfg['project_path']+'/'+'model/temporal_model/best_model'):
            os.mkdir(cfg['project_path']+'/'+'model/temporal_model')
            os.mkdir(cfg['project_path']+'/model/'+'temporal_model/best_model')
            os.mkdir(cfg['project_path']+'/model/'+'temporal_model/losses')
            
    # make sure torch uses cuda for GPU computing
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Using CUDA")
        print('GPU active:',torch.cuda.is_available())
        print('GPU used:',torch.cuda.get_device_name(0)) 
    else:
        print("CUDA is not working!")
    
    """ HYPERPARAMTERS """
    CUDA = use_gpu
    SEED = 19
    TRAIN_BATCH_SIZE = 256
    TEST_BATCH_SIZE = 64
    EPOCHS = cfg['Epochs_temporal']
    ZDIMS = cfg['ZDIMS_temporal']
    BETA  = 1
    LEARNING_RATE = cfg['Learning_rate_temporal']
    NUM_FEATURES = cfg['num_features']
    TEMPORAL_WINDOW = cfg['temporal_window']
    FUTURE_STEPS = TEMPORAL_WINDOW - cfg['future_steps']
    KL_START = 3
    ANNEALTIME = 8
    BEST_LOSS = 999999
    convergence = 0
    print('Latent Dimensions: %d, Beta: %d, lr: %.4f' %(ZDIMS, BETA, LEARNING_RATE))
    
    # simple logging of diverse losses
    train_losses = []
    test_losses = []
    kmeans_losses = []
    kl_losses = []
    weight_values = []
    mse_losses = []
    fut_losses = []
    
    anneal_function = cfg['anneal_function']
    
    torch.manual_seed(SEED)
    
    if CUDA:
        torch.cuda.manual_seed(SEED)
        model = RNN_VAE(TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES,FUTURE_DECODER,FUTURE_STEPS).cuda()
    else:
        model = RNN_VAE(TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES,FUTURE_DECODER,FUTURE_STEPS)
        
    if pretrained:
#        model.load_state_dict(torch.load('./model/temporal/pretrained/vae_resnet_state.pkl'))
        if os.path.exists(cfg['project_path']+'/'+'model/'+folder+'/best_model/'+model_name+'_'+cfg['Project']+'.pkl'):
            print("Load pretrained Model: %s" %model_name)
            model.load_state_dict(torch.load(cfg['project_path']+'/'+'model/'+folder+'/best_model/'+model_name+'_'+cfg['Project']+'.pkl'))
            KL_START = 1
            ANNEALTIME = 1
            
    """ DATASET """
    if spatial_features:
        trainset = SPATIAL_FEATURES(cfg['project_path']+'data/train/temporal/', data='spatiotemporal_train_full.npy', train=True, temporal_window=TEMPORAL_WINDOW)
        testset = SPATIAL_FEATURES(cfg['project_path']+'data/train/temporal/', data='spatiotemporal_test_full.npy', train=False, temporal_window=TEMPORAL_WINDOW)
     
    else:
        trainset = SEQUENCE_DATASET(cfg['project_path']+'data/train/temporal/', data='sequence_train_full.npy', train=True, temporal_window=TEMPORAL_WINDOW)
        testset = SEQUENCE_DATASET(cfg['project_path']+'data/train/temporal/', data='sequence_test_full.npy', train=False, temporal_window=TEMPORAL_WINDOW)
        
    train_loader = Data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = Data.DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle=True, drop_last=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, amsgrad=True)
    
    scheduler = StepLR(optimizer, step_size=300, gamma=0.2, last_epoch=-1)

    for epoch in range(1,EPOCHS):
        print('Epoch: %d' %epoch)
        print('Train: ')
        weight, train_loss, km_loss, kl_loss, mse_loss, fut_loss = train(train_loader, epoch, model, 
                                                                         optimizer, anneal_function, 
                                                                         BETA, KL_START, ANNEALTIME, 
                                                                         TEMPORAL_WINDOW, 
                                                                         FUTURE_DECODER, FUTURE_STEPS, 
                                                                         scheduler)
        
        print('Test: ')
        current_loss, test_loss, test_list = test(test_loader, epoch, model, optimizer, 
                                                  BETA, weight, TEMPORAL_WINDOW, FUTURE_DECODER, 
                                                  FUTURE_STEPS)
        
        for param_group in optimizer.param_groups:
            print('lr: {}'.format(param_group['lr']))
        # logging losses
#        train_iter_loss = train_iter_loss + train_list
#        test_iter_loss = test_iter_loss + test_list
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        kmeans_losses.append(km_loss)
        kl_losses.append(kl_loss)
        weight_values.append(weight)
        mse_losses.append(mse_loss)
        fut_losses.append(fut_loss)
        
        if epoch == 100 and debug == True:
            convergence = 1000
            
#        # save best model
#        if weight > 0.99 and current_loss <= BEST_LOSS:
#            BEST_LOSS = current_loss
#            print("Saving model!\n")
#            torch.save(model.state_dict(), cfg['project_path']+'/'+'model/'+folder+'/best_model'+'/'+model_name+'_'+cfg['Project']+'.pkl')
#            convergence = 0
#        else:
#            convergence += 1
        if epoch % 50 == 0:
            print("Saving model!\n")
            torch.save(model.state_dict(), cfg['project_path']+'/'+'model/'+folder+'/best_model'+'/snapshots/'+model_name+'_'+cfg['Project']+'_epoch_'+str(epoch)+'.pkl')
        
        if convergence > cfg['model_convergence']:
            print('Model converged. Please check your model with vame.evaluate_model(). \n'
                  'You can also re-run vame.temporal() to further improve your model. \n'
                  'Hint: Set "model_convergence" in your config.yaml to a higher value. \n'
                  '\n'
                  'Next: \n'
                  'Use vame.segment_behavior() to identify behavioral motifs in your dataset!')
            #return
            break
        
        
        # save logged losses
#        np.save(cfg['project_path']+'/'+'model/'+folder+'/losses'+'/train_iter_losses'+model_name, train_iter_loss)
#        np.save(cfg['project_path']+'/'+'model/'+folder+'/losses'+'/test_iter_losses'+model_name, test_iter_loss)
        np.save(cfg['project_path']+'/'+'model/'+folder+'/losses'+'/train_losses'+model_name, train_losses)
        np.save(cfg['project_path']+'/'+'model/'+folder+'/losses'+'/test_losses'+model_name, test_losses)
        np.save(cfg['project_path']+'/'+'model/'+folder+'/losses'+'/kmeans_losses'+model_name, kmeans_losses)
        np.save(cfg['project_path']+'/'+'model/'+folder+'/losses'+'/kl_losses'+model_name, kl_losses)
        np.save(cfg['project_path']+'/'+'model/'+folder+'/losses'+'/weight_values'+model_name, weight_values)
        np.save(cfg['project_path']+'/'+'model/'+folder+'/losses'+'/mse_losses'+model_name, mse_losses)
        np.save(cfg['project_path']+'/'+'model/'+folder+'/losses'+'/fut_losses'+model_name, fut_losses)


    if convergence < cfg['model_convergence']:
        print('Model seemed to have not reached convergence. You may want to check your model \n'
              'with vame.evaluate(). If your satisfied you can continue with: \n\n'
               '- Use vame.segment_behavior() to identify behavioral motifs in your dataset!\n\n'
               'OPTIONAL: You can re-run vame.temporal().')













