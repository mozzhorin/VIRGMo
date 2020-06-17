# -*- coding: utf-8 -*-
"""
Variational inference for graphon.
"""

import time
import warnings
import pickle
import copy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch.autograd import Variable
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.beta import Beta
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from graph_models import WDCSBM, DCSBM, SBM, EdgesDataset, WCRG, Graphon, WeightedGraphon
from vi_sbm import VI_RG

###############################################################################

class MLP(nn.Module):
    
    def __init__(self, units=[25,1,25], sigmoid=True):                                   
        super(MLP, self).__init__() 
        self.sigmoid = sigmoid
        self.layer1 = nn.Linear(1, units[0])
        self.layer2 = nn.Linear(units[0], units[1])
        self.layer3 = nn.Linear(units[1]*2, units[2])
        self.layer4 = nn.Linear(units[2], 1)
        
    def forward(self, x, y):
        x = self.layer2(torch.tanh(self.layer1(x)))
        y = self.layer2(torch.tanh(self.layer1(y)))
        z = torch.stack((x,y),dim=-1)
        if self.sigmoid:
            z = torch.sigmoid(self.layer4(self.layer3(z)))
        else:
            z = self.layer4(self.layer3(z))
        return z.squeeze(-1).squeeze(-1)
    
    def l2(self):
        s = 0
        for p in self.parameters():
            s += p.pow(2).sum()
        return s
        
class VI_Graphon(VI_RG):
    '''
    Variational inference for graphon models.
    
    Parameters:
    
    num_classes (int): number of classes K
    num_nodes (int): number of nodes N    
        
    EXAMPLE: 
    
    N = 50
    g = Graphon()
    u,A = g.generate(N)
    g.show(sorted=True)
    order = A.sum(dim=0).argsort()
    A_ordered = A[order,:][:,order]
    dataloader = DataLoader(EdgesDataset(A_ordered), 
                            batch_size=20, shuffle=True, num_workers=0)
    degree = A_ordered.sum(dim=1)
    truth = g.W(u[order],u[order])
    model = VI_Graphon(num_nodes=N, units=[25,1,25], num_samples=20,
                       init_values={'us':None, 'degrees':degree})
    model.train(dataloader, epochs=30, lrs=0.1)
    model.show_predictions()
    model.show_error(truth)
    model.show_us()
    '''
    def __init__(self, num_nodes=50,
                 units=[25,1,25], 
                 priors = {'u_p':None},
                 init_values={'us':None, 'degrees':None},                
                 num_samples=10,
                 train_scaling=0.05,
                 device=None):
        ''' Initialize the model.
        
        ARGUMENTS:
        
        num_classes (int): number of classes K
        num_nodes (int): number of nodes N
        
        '''                           
        super(VI_Graphon, self).__init__()        
        
        self.num_nodes = num_nodes 
        self.num_samples = num_samples
        self.units = units
        self.init_values = init_values
        self.train_scaling = train_scaling
        self.multi_results = None
        self.state_dicts = None
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cpu")
        
        self.params_reset()          
        
        if priors['u_p'] is None:
            u_p = torch.ones([self.num_nodes, 2])
        else:
            u_p = priors['u_p']        
        self.u_p = Variable(u_p, requires_grad=False).to(self.device)
        
    def params_reset(self):
        ''' Reset parameters of the variational distribution from the 
        init_values dictionary or with random values.'''
        
        if self.init_values['us'] is None:
            us = torch.rand([self.num_nodes, 2]).to(self.device)
        else:
            us = self.init_values['us'].to(self.device)
        # Initialize smooth monotone 'u' proportional to the node's degree     
        if not self.init_values['degrees'] is None:
           c = lambda x: 3 - 2*torch.tanh(x*6-3).abs()
           a = lambda m: (5*m+1)*c(m)
           b = lambda m: (-5*m+6)*c(m)
           deg_norm = self.init_values['degrees']/self.init_values['degrees'].max()
           us = torch.stack((a(deg_norm), b(deg_norm)), dim=1).to(self.device)
        
        self.us = nn.Parameter(us)
        self.add_module("W", MLP(self.units))
    
    def constrained_params(self):
        ''' Return constrained posterior parameters. '''
        return torch.exp(self.us)    
        
    def elbo(self, idx1, idx2, weights):
        batch = len(weights)
        u_x = self.constrained_params()
        u_samples = Beta(u_x[:,0],u_x[:,1])\
                .rsample([self.num_samples]).to(self.device)
        edges = torch.where(weights>0, 
                            torch.ones(weights.size()).to(self.device), 
                            torch.zeros(weights.size()).to(self.device))
        edges_flat = torch.flatten(torch.ones([self.num_samples,batch])*edges)
        pred_W = self.W(u_samples[:,idx1].flatten().unsqueeze(-1), 
                      u_samples[:,idx2].flatten().unsqueeze(-1))        
        lpdf = Bernoulli(pred_W).log_prob(edges_flat)
        elbo = - self.train_scaling /self.num_nodes * diriKL(u_x[idx1], self.u_p[idx1]).sum()
        elbo += lpdf.view(self.num_samples,batch).mean(dim=0).sum()        
        return elbo

    def qmean(self):
        u_x = self.constrained_params()
        u_mean = Beta(u_x[:,0],u_x[:,1]).mean.detach().clone()
        return u_mean        
    
    def get_W(self):
        ''' Returns predicitions for W.
        
        ARGUMENTS:
        '''
        predictions = torch.zeros([self.num_nodes,self.num_nodes]).to(self.device)
        u_mean = self.qmean()
        u_ij = torch.zeros([self.num_nodes,self.num_nodes,2])
        u_ij[:,:,0] = u_mean
        u_ij[:,:,1] = u_mean.unsqueeze(0).t()
        predictions = self.W(u_ij[:,:,0].flatten().unsqueeze(-1), 
                             u_ij[:,:,1].flatten().unsqueeze(-1))        
        return predictions.detach().view(self.num_nodes,self.num_nodes)
    
    def show_predictions(self):
        ''' Shows predicitions.
        
        ARGUMENTS:
        '''
        predictions = self.get_W()
        plt.title("Predictions")
        plt.imshow(predictions.numpy())
        plt.colorbar()
        plt.show()
        
    def get_error(self, truth):
        predictions = self.get_W()
        return predictions-truth
        
    def show_error(self, truth, threshold=0):
        error = self.get_error(truth)
        err_treshold = torch.where(error.abs() < threshold, 
                                   torch.zeros(error.size()),
                                   error)
        plt.title("Error")
        plt.imshow(err_treshold.numpy())
        plt.colorbar()
        plt.show()
        
    def summary(self, truth=None, threshold=0.1):
        self.show_predictions()
        if not truth is None:
            self.show_error(truth, threshold)
        self.show_us()
        
        
    def show_us(self, us=None):
        if us is None:
            us = self.constrained_params().detach().clone()
        x = torch.arange(0, 1.02, 0.02)
        for i in range(len(us)):
            plt.plot(x.numpy(), Beta(us[i,0], us[i,1]).log_prob(x).exp().numpy())
        plt.show()
        
    def multi_train(self, dataloader, optimizer='rmsprop', momentum=None,
                    lrs=0.1, epochs=10, trials=10,
                    init_states=None):
        ''' Fit the model several times. Used when the model has many local
        optimas and the quality of fit highly depends on the initial values of
        the variational distribution's parameters.
        
        ARGUMENTS:
        
        dataloader (torch.utils.data.DataLoader):
            dataloader to iterate over the list of edges.
        optimizer (str, {rmsprop, adagrad, adam, asgd, sgd}): 
            optimizer's name
        lrs (float or list): learning rates; if it is a list of floats, 
            the model is trained with each learning rate for the corresponding 
            number of epochs from the epochs' list or with each learning rate 
            for the same number of epochs, if 'epochs' argument is an integer.            
        epochs (int or list): number of training epochs for each learning rate
            or for the corresponding learning rate from theirs list.
        momentum (float): momentum factor.
        trials (int): number of training trials.
        init_states (list of OrderedDict, size: trials): 
            state dictionaries from torch.nn.Module.state_dict() for further 
            training. If None, trials initialized from self.init_values.       
        
        '''
        us, ws, mus, taus, losses, state_dicts = [],[],[],[],[],[]
        
        for i in range(trials):
            if init_states is None:
                self.params_reset()
            else:
                self.load_state_dict(init_states[i])
            print('>>>>>>> Training iteration #%d \n' % (i+1))
            try:
		self.train(dataloader, optimizer=optimizer, epochs=epochs, lrs=lrs,
		               momentum=momentum)
		us.append(self.qmean())
		ws.append(self.get_W())
		try:
		    mus.append(self.get_mu())
		    taus.append(self.get_tau())
		except:
		    pass
		losses.append(self.loss_list)
		state_dicts.append(self.state_dict())
            except:
            	pass
            
        us=torch.stack(us)
        losses = torch.stack(losses)
        ws=torch.stack(ws)
        if len(mus):
            mus=torch.stack(mus)
            taus=torch.stack(taus)
            self.multi_results = [us, ws, mus, taus, losses]
        else:
            self.multi_results = [us, ws, losses]
            
        self.state_dicts = state_dicts
        
###############################################################################
        
class VI_WeightedGraphon(VI_Graphon):
    '''
    Variational inference for graphon models.
    
    Parameters:
    
    num_classes (int): number of classes K
    num_nodes (int): number of nodes N    
        
    EXAMPLE: 
    
    N = 50
    g = WeightedGraphon()
    u,A = g.generate(N)
    g.show(sorted=True)
    order = A.sum(dim=0).argsort()
    A_ordered = A[order,:][:,order]
    dataloader = DataLoader(EdgesDataset(A_ordered), 
                            batch_size=20, shuffle=True, num_workers=0)
    degree = A_ordered.sum(dim=1)
    truth = g.get_W(u[order])
    model = VI_WeightedGraphon(num_nodes=N, units=[25,1,25], num_samples=20,
                       init_values={'us':None, 'degrees':degree})
    model.train(dataloader, epochs=30, lrs=0.1)
    model.show_predictions()
    model.show_error(truth)
    model.show_us()
    '''
    def __init__(self, num_nodes=50,
                 units=[25,1,25], 
                 priors = {'u_p':None},
                 init_values={'us':None, 'degrees':None},                
                 num_samples=10,
                 train_scaling=0.05,
                 penalty=1e-4,
                 device=None):
        ''' Initialize the model.
        
        ARGUMENTS:
        
        num_classes (int): number of classes K
        num_nodes (int): number of nodes N
        
        '''                           
        super(VI_WeightedGraphon, self).__init__(
            num_nodes, units, priors, init_values, 
            num_samples, train_scaling, device) 
        self.penalty = penalty
        
                
    def params_reset(self):
        ''' Reset parameters of the variational distribution from the 
        init_values dictionary or with random values.'''
        
        super(VI_WeightedGraphon, self).params_reset()
        self.add_module("mu", MLP(self.units, sigmoid=False))
        self.add_module("tau", MLP(self.units, sigmoid=False))
        
    def elbo(self, idx1, idx2, weights):
        batch = len(weights)        
        u_x = self.constrained_params()
        u_samples = Beta(u_x[:,0],u_x[:,1]).rsample([self.num_samples]).to(self.device)
        elbo = - self.train_scaling /self.num_nodes * diriKL(u_x[idx1], self.u_p[idx1]).sum()
        edges = torch.where(weights>0, 
                            torch.ones(weights.size()).to(self.device), 
                            torch.zeros(weights.size()).to(self.device))
        edges_flat = torch.flatten(torch.ones([self.num_samples,batch])*edges)
        pred_W = self.W(u_samples[:,idx1].flatten().unsqueeze(-1), 
                      u_samples[:,idx2].flatten().unsqueeze(-1))        
        lpdf_W = Bernoulli(pred_W).log_prob(edges_flat)
        elbo += lpdf_W.view(self.num_samples,batch).mean(dim=0).sum()         
        
        weights_ = weights[weights.nonzero()].squeeze(-1)
        batch_ = len(weights_) 
        idx1_ = idx1[weights.nonzero()].squeeze(-1)
        idx2_ = idx2[weights.nonzero()].squeeze(-1) 
        weights_flat = torch.flatten(torch.ones([self.num_samples,batch_])*weights_)
        pred_mu_x = self.mu(u_samples[:,idx1_].flatten().unsqueeze(-1), 
                          u_samples[:,idx2_].flatten().unsqueeze(-1)).exp()
        pred_tau_x = self.tau(u_samples[:,idx1_].flatten().unsqueeze(-1), 
                            u_samples[:,idx2_].flatten().unsqueeze(-1)).exp()            
        log_w = weights_flat.log() 
        lpdf_X = - 0.5*torch.tensor(np.pi*2).log().to(self.device) \
                 - log_w \
                 + 0.5*pred_tau_x.log() \
                 - 0.5*pred_tau_x*(log_w - pred_mu_x)**2
        elbo += lpdf_X.view(self.num_samples,batch_).mean(dim=0).sum()
        elbo -= self.penalty*(self.mu.l2()+self.tau.l2())
        return elbo
        
    def get_mu(self):
        ''' Returns predicitions for mu.
        
        ARGUMENTS:
        '''
        predictions = torch.zeros([self.num_nodes,self.num_nodes]).to(self.device)
        u_mean = self.qmean()
        u_ij = torch.zeros([self.num_nodes,self.num_nodes,2])
        u_ij[:,:,0] = u_mean
        u_ij[:,:,1] = u_mean.unsqueeze(0).t()
        predictions = self.mu(u_ij[:,:,0].flatten().unsqueeze(-1), 
                             u_ij[:,:,1].flatten().unsqueeze(-1))        
        return predictions.detach().view(self.num_nodes,self.num_nodes)
        
    def get_tau(self):
        ''' Returns predicitions for tau.
        
        ARGUMENTS:
        '''
        predictions = torch.zeros([self.num_nodes,self.num_nodes]).to(self.device)
        u_mean = self.qmean()
        u_ij = torch.zeros([self.num_nodes,self.num_nodes,2])
        u_ij[:,:,0] = u_mean
        u_ij[:,:,1] = u_mean.unsqueeze(0).t()
        predictions = self.tau(u_ij[:,:,0].flatten().unsqueeze(-1), 
                             u_ij[:,:,1].flatten().unsqueeze(-1))        
        return predictions.detach().view(self.num_nodes,self.num_nodes)
    
    def show_predictions(self):
        ''' Shows predicitions.
        
        ARGUMENTS:
        '''
        W = self.get_W()
        M = self.get_mu().exp()
        T = self.get_tau().exp()
        plt.title("Predictions W")
        plt.imshow(W.numpy())
        plt.colorbar()
        plt.show()
        plt.title("Predictions mu")
        plt.imshow(M.numpy())
        plt.colorbar()
        plt.show()
        plt.title("Predictions tau")
        plt.imshow(T.numpy())
        plt.colorbar()
        plt.show()
        
    
