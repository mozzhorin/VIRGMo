# -*- coding: utf-8 -*-
"""
Variational inference for hyperbilic random graph models.
"""
import time
#import itertools
#import warnings
#import pickle
#import copy
#import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import float64 as f64
from torch.autograd import Variable
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.beta import Beta
#from torch.distributions.normal import Normal
from torch.distributions.log_normal import LogNormal
from torch.utils.data import Dataset, DataLoader
from torch.distributions.kl import kl_divergence
from utils import diriKL, normKL, c2d, bad_tensor, warn_tensor#, hyperdist, p_hd
from graph_models import EdgesDataset, HRG

from distributions.von_mises_fisher import VonMisesFisher
from distributions.radius import Radius

softmax = torch.nn.Softmax(dim=0)

class VI_HRG(torch.nn.Module):
    '''
    Variational inference for hyperbilic random graph models.
    
    PARAMETERS:    
    
    num_nodes (int): number of nodes N
    
    rs (torch.Tensor, size: N*2): posterior r-coordinate of each node.
    phis (torch.Tensor, size: N*2): posterior phi-coordinate of each node.
    R (torch.Tensor, size: 2): posterior R.
    alpha (torch.Tensor, size: 2): posterior alpha.
    T (torch.Tensor, size: 2): posterior T.
    
    R_p (torch.Tensor, size: 2): prior R.
    alpha_p (torch.Tensor, size: 2): prior alpha.
    T_p (torch.Tensor, size: 2): prior T.
    
    
    N = 75
    G = HRG(R=torch.tensor([5.0]).to(torch.double),
            alpha=1.1,
            T=0.1)
    r, theta, A = G.generate(N)
    G.show()
    G.plot()
    dataloader = DataLoader(EdgesDataset(A), 
                            batch_size=64, shuffle=True, num_workers=0)
    vi = VI_HRG(75,20)
    vi.train(dataloader)
    
    '''
    def __init__(self, num_nodes=50, num_samples=20, 
                 priors={'R_p':None, 
                         'T_p':None,
                         'alpha_p':None},
                 init_values={'rs_loc':None,
                              'rs_scale':None, 
                              'phis_loc':None,
                              'phis_scale':None, 
                              'R_loc':None,
                              'R_scale':None, 
                              'T':None,
                              'alpha_loc':None,
                              'alpha_scale':None},
                 device=None):
        ''' Initialize the model.
        
        ARGUMENTS:
        
        num_nodes (int): number of nodes N
        priors (dict of torch.float): priors
        init_values (dict of torch.float): initial values of the variational 
            distribution's parameters.
        '''
        super(VI_HRG, self).__init__()        
        self.num_nodes = num_nodes
        self.num_samples = num_samples
        self.init_values = init_values
        self.dtype = f64
        self.epsilon = 1e-12  # Keeps some params away from extream values
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cpu")
        # Initialize parameters of variational distribution      
        self.params_reset()        
        
        # Initialize the priors fron the priors' dictionary or with 
        # default values
        
        if priors['R_p'] is None:
            R_p = torch.tensor([0., 1.]).to(self.device).to(self.dtype)
        else:
            R_p = priors['R_p'].to(self.device).to(self.dtype) 
            
        if priors['T_p'] is None:
            T_p = torch.ones([2]).to(self.device).to(self.dtype)
        else:
            T_p = priors['T_p'].to(self.device).to(self.dtype)
        
        if priors['alpha_p'] is None:
            alpha_p = torch.tensor([0., 1.]).to(self.device).to(self.dtype)
        else:
            alpha_p = priors['alpha_p'].to(self.device).to(self.dtype)
            
        self.R_p = Variable(R_p, requires_grad=False).to(self.device)
        self.T_p = Variable(T_p, requires_grad=False).to(self.device)
        self.alpha_p = Variable(alpha_p, requires_grad=False).to(self.device)
    
    def params_reset(self):
        ''' Reset parameters of the variational distribution from the 
        init_values dictionary or with random values.'''
        
        if self.init_values['rs_loc'] is None:
            rs_loc = torch.rand([self.num_nodes])
        else:
            rs_loc = self.init_values['rs_loc']
            
        if self.init_values['rs_scale'] is None:
            rs_scale = torch.rand([self.num_nodes])
        else:
            rs_scale = self.init_values['rs_scale']
            
        if self.init_values['phis_loc'] is None:
            phis_loc = torch.rand([self.num_nodes,3])
        else:
            phis_loc = self.init_values['phis_loc']
            
        if self.init_values['phis_scale'] is None:
            phis_scale = torch.rand([self.num_nodes])
        else:
            phis_scale = self.init_values['phis_scale']
            
        if self.init_values['R_loc'] is None:
            R_loc = torch.rand()
        else:
            R_loc = self.init_values['R_loc']
            
        if self.init_values['R_scale'] is None:
            R_scale = torch.rand()
        else:
            R_scale = self.init_values['R_scale']
            
        if self.init_values['T'] is None:
            T = torch.rand([2])
        else:
            T = self.init_values['T']
            
        if self.init_values['alpha_loc'] is None:
            alpha_loc = torch.rand()
        else:
            alpha_loc = self.init_values['alpha_loc']
            
        if self.init_values['alpha_scale'] is None:
            alpha_scale = torch.rand()
        else:
            alpha_scale = self.init_values['alpha_scale']
            
        self.rs_loc = torch.nn.Parameter(rs_loc.to(self.device).to(self.dtype))
        self.rs_scale = torch.nn.Parameter(rs_scale.to(self.device).to(self.dtype))
        self.phis_loc = torch.nn.Parameter(phis_loc.to(self.device).to(self.dtype))
        self.phis_scale = torch.nn.Parameter(phis_scale.to(self.device).to(self.dtype))
        self.R_loc = torch.nn.Parameter(R_loc.to(self.device).to(self.dtype))
        self.R_scale = torch.nn.Parameter(R_scale.to(self.device).to(self.dtype))
        self.T = torch.nn.Parameter(T.to(self.device).to(self.dtype))
        self.alpha_loc = torch.nn.Parameter(alpha_loc.to(self.device).to(self.dtype))
        self.alpha_scale = torch.nn.Parameter(alpha_scale.to(self.device).to(self.dtype))
        
        
    def constrained_params(self):
        ''' Return constrained posterior parameters. '''
        return (self.rs_loc,
                torch.exp(self.rs_scale),
                self.phis_loc,
                torch.exp(self.phis_scale),
                self.R_loc,
                torch.exp(self.R_scale),
                torch.exp(self.T),
                self.alpha_loc,
                torch.exp(self.alpha_scale))
                
#    def omega_approx(self, R_x, T_x, alpha_x, idx1, idx2, weights):
#        ''' Returns part of ELBO (see the documentation).
#        Estimated through sampling.
#        
#        ARGUMENTS:        
#        
#        B_x (torch.Tensor, size: K*K*2): constrained Bs.
#        eta_x (torch.Tensor, size: K*L): constrained etas' batch.
#        delta_x (torch.Tensor, size: N*2): constrained deltas.
#        idx1 (torch.int, size: L): start nodes.
#        idx2 (torch.int, size: L): finish nodes.            
#        weights (torch.float, size: L): edges' weights.
#        '''
##        delta_i = Normal(delta_x[idx1,0],delta_x[idx1,1])\
##                    .rsample([self.num_samples]).to(self.device)
##        delta_j = Normal(delta_x[idx2,0],delta_x[idx2,1])\
##                    .rsample([self.num_samples]).to(self.device)     
##        B_ij = Beta(B_x[:,:,0],B_x[:,:,1])\
##                    .rsample([self.num_samples]).to(self.device)
##        sig = torch.sigmoid(delta_i+delta_j)
##        B = sig.unsqueeze(-1).unsqueeze(-1) * B_ij.unsqueeze(1)
##        ElogB = B.log().mean(dim=0)
##        ElogB_ = (1. - B).log().mean(dim=0)
##        edges = torch.where(weights>0, 
##                            torch.ones(weights.size()).to(self.device), 
##                            torch.zeros(weights.size()).to(self.device))
##        log_pA = edges.unsqueeze(-1).unsqueeze(-1)*ElogB \
##               + (1.-edges).unsqueeze(-1).unsqueeze(-1)*ElogB_            
##        temp = torch.mul(eta_x[:,idx1].unsqueeze(0),
##                         eta_x[:,idx2].unsqueeze(1)).transpose(0,2)
#        L = len(weights)        
#        
#        r_i = Radius()
#        p_dist = 1
#        
#        log_pA = edges.unsqueeze(-1).unsqueeze(-1)*p_dist \
#               + (1.-edges).unsqueeze(-1).unsqueeze(-1)*p_dist_                  
#        return torch.mul(log_pA, temp).sum()
                
    def elbo(self, idx1, idx2, weights):
        ''' Return evidence lower bound (ELBO) calculated for a nodes batch 
        of size L; also the loss for the training.
        
        ARGUMENTS:
        
        idx1 (torch.int, size: L): start nodes.
        idx2 (torch.int, size: L): finish nodes.
        weights (torch.float, size: L): edges weights.
        
        '''
        
        L = len(weights)   # Batch size
        r_x_loc, r_x_scale, phi_x_loc, phi_x_scale, R_x_loc, R_x_scale, T_x, \
            alpha_x_loc, alpha_x_scale = self.constrained_params() 
        
        edges = torch.where(weights>0, 
                            torch.ones(weights.size()).to(self.device).to(self.dtype), 
                            torch.zeros(weights.size()).to(self.device).to(self.dtype))
        
        R_q = LogNormal(R_x_loc, R_x_scale)
        T_q = Beta(T_x[0], T_x[1])
        alpha_q = LogNormal(alpha_x_loc, alpha_x_scale)
                    
        loop_count = 0
        while True:
            loop_count += 1
            R_samples = R_q.rsample([L]).squeeze(-1).to(self.device).to(self.dtype)
            T_samples = T_q.rsample([L]).squeeze(-1).to(self.device).to(self.dtype)
            alpha_samples = alpha_q.rsample([L]).squeeze(-1).to(self.device).to(self.dtype)
        
            r_i_q = Radius(r_x_loc[idx1], r_x_scale[idx1], R_samples)
            r_i_samples = r_i_q.rsample([self.num_samples]).to(self.device).to(self.dtype)
            r_j_q = Radius(r_x_loc[idx2], r_x_scale[idx2], R_samples)
            r_j_samples = r_j_q.rsample([self.num_samples]).to(self.device).to(self.dtype)
            alpha_r_i = (alpha_samples*r_i_samples).sinh().log()
            alpha_R = ((alpha_samples*R_samples).cosh()-1).log()
            if not (bad_tensor(warn_tensor(alpha_r_i, 'alpha_ri')) \
                or bad_tensor(warn_tensor(alpha_R, 'alpha_R'))):
                break
            if loop_count > 1000:
                raise Exception('Infinite loop!!!')
        
        phi_i_q = VonMisesFisher(phi_x_loc[idx1], phi_x_scale[idx1].unsqueeze(dim=-1))
        phi_i_samples = phi_i_q.rsample(self.num_samples).to(self.device).to(self.dtype)#[:,:,:2]
        phi_j_q = VonMisesFisher(phi_x_loc[idx2], phi_x_scale[idx2].unsqueeze(dim=-1))
        phi_j_samples = phi_j_q.rsample(self.num_samples).to(self.device).to(self.dtype)#[:,:,:2]
        #print(phi_i_samples.shape)

        arcosh = lambda x: (x + (x**2 - 1).sqrt()).log()
        hyperdist = lambda rx,ry,fx,fy: arcosh(rx.cosh()*ry.cosh() - rx.sinh()*ry.sinh()*(fx-fy).cos())
        p_hd_ = lambda d,R,T: (1+((d-R)/(2*T)).exp()).reciprocal()
        phd = lambda d,R,T: 0.5 + 0.5*(-4*(d-R)/T).tanh()
        
        dist = hyperdist(r_i_samples, r_j_samples, c2d(phi_i_samples), c2d(phi_j_samples))
        p_ = phd(dist, R_samples, T_samples)
        p_dist = torch.clamp(p_, min=self.epsilon, max=1.-self.epsilon)
        prob_edges = Bernoulli(p_dist).log_prob(edges).mean(dim=0)
        #print(dist)
        #print('>>', p_dist)
        #E_log_p_dist = p_dist.log().mean(dim=0)
        #print('>>>',E_log_p_dist)
        #E_log_p_dist_ = (1-p_dist).log().mean(dim=0)
        #log_pA = edges*E_log_p_dist + (1.-edges)*E_log_p_dist_
        #print('log_pA',log_pA)
        # Calculate and sum different parts of ELBO
        count = 0
        elbo = 0
        #elbo = -1/self.num_nodes**2 * normKL(R_x, self.R_p).double()
        elbo -= 1/self.num_nodes**2 * \
            kl_divergence(R_q, LogNormal(self.R_p[0], self.R_p[1])).double()
#        print('ELBO', str(count), '>>', str(elbo)); count +=1
        #elbo -= 1/self.num_nodes**2 * normKL(alpha_x, self.alpha_p).double()
        elbo -= 1/self.num_nodes**2 * \
            kl_divergence(alpha_q, LogNormal(self.alpha_p[0], self.alpha_p[1])).double()
#        print('ELBO', str(count), '>>', str(elbo)); count +=1
        #elbo -= 1/self.num_nodes**2 * diriKL(T_x, self.T_p).double()
        elbo -= 1/self.num_nodes**2 * \
            kl_divergence(T_q, Beta(self.T_p[0], self.T_p[1])).double()
#        print('ELBO', str(count), '>>', str(elbo)); count +=1
        elbo += warn_tensor(prob_edges, 'log_pA').sum()
#        print('ELBO', str(count), '>>', str(elbo)); count +=1
        
        elbo += 1/self.num_nodes * alpha_r_i.mean(dim=0).sum() 
#        print('ELBO', str(count), '>>', str(elbo)); count +=1

        elbo += 1/self.num_nodes * alpha_samples.log().mean()
#        print('ELBO', str(count), '>>', str(elbo)); count +=1
        elbo -= 1/self.num_nodes * alpha_R.mean()
#        print('ELBO', str(count), '>>', str(elbo)); count +=1
        elbo -= 1/self.num_nodes * r_i_q.log_prob(r_i_samples).mean(dim=0).sum()
#        print('ELBO', str(count), '>>', str(elbo)); count +=1
        elbo -= 1/self.num_nodes * phi_i_q.log_prob(phi_i_samples).mean(dim=0).sum()
#        print('ELBO', str(count), '>>', str(elbo)); count +=1
        return elbo
        
    def train_(self, epoch_num):
        ''' Fit the variational distribution for one epoch.
        
        ARGUMENTS:
        
        epoch_num (int): the epoch's number for printing it out.        
        '''
        t1 = time.time()  # Measure the training time
        total_loss = 0
        for idx1, idx2, data in self.dataloader:
            idx1, idx2, data = idx1.to(self.device), idx2.to(self.device), data.to(self.device)
            loss = - self.elbo(idx1, idx2, data)
            self.optimizer.zero_grad()
            loss.backward(retain_graph=False)
            self.optimizer.step()
            total_loss += loss
            if data.sum()<=0:
                warnings.warn('Batch is empty! Your graph is to sparse, increase the batch size!')            
        t2 = time.time()
        tl = total_loss.cpu().data.numpy().item()
        print('Epoch %d | LR: %.2f | Total loss: %.2f | Epoch time %.2f'\
                  % (epoch_num+1, self.optimizer.lr, tl, (t2-t1)))
        return tl
    
    def train(self, dataloader, optimizer='rmsprop', 
              lrs=0.1, epochs=10, momentum=None):
        ''' Fit the variational distribution for a number of epochs.
        
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
        momentum (float): momentum factor 
        
        IMPORTANT: 
            the number of elements in lrs' end epochs' lists must be the same.
        '''
        self.dataloader = dataloader
        self.lrs = lrs
        self.epochs = epochs
        
        # Set the optimizer
        if optimizer=='rmsprop':
            self.optimizer = torch.optim.RMSprop(self.parameters())   
        elif optimizer=='adagrad':
            self.optimizer = torch.optim.Adagrad(self.parameters())
        elif optimizer=='adam':
            self.optimizer = torch.optim.Adam(self.parameters())
        elif optimizer=='asgd':
            self.optimizer = torch.optim.ASGD(self.parameters())
        elif optimizer=='sgd':
            self.optimizer = torch.optim.SGD(self.parameters())
        
        # Set the momentum if specified
        if momentum is not None:
            self.optimizer.momentum = momentum
        
        # Total loss for traning monitoring
        self.loss_list = [] 
        
        # Training loop
        print('>>>>>>>>>>>> Start training...')
        epoch_counter = 0
        # Check if lrs is a float, then run only one loop
        if isinstance(self.lrs, float) or isinstance(self.lrs, int):
            self.optimizer.lr=self.lrs
            for e in range(self.epochs):
                   curr_loss = self.train_(epoch_counter)
                   epoch_counter +=1
                   self.loss_list.append(curr_loss)
        else:
            # If lrs is a list and epochs is an integer, train the model for 
            # the same number of epochs with each lr
            if isinstance(self.epochs, int):
                for lr in self.lrs:
                    self.optimizer.lr=lr
                    for e in range(self.epochs):
                       curr_loss = self.train_(epoch_counter)
                       epoch_counter +=1
                       self.loss_list.append(curr_loss)
            # If both lrs and epochs are lists, train the model with 
            # corresponding lr and epoch
            else:
                for l in range(len(self.lrs)):
                    self.optimizer.lr=self.lrs[l]
                    for e in range(self.epochs[l]):
                            curr_loss = self.train_(epoch_counter)
                            epoch_counter +=1
                            self.loss_list.append(curr_loss)                 
        self.loss_list = torch.tensor(self.loss_list)
        print('>>>>>>>>>>>> Training is finished.\n')
    
