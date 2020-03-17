# -*- coding: utf-8 -*-
"""
Variational inference for hyperbilic random graph models.
"""
import time
#import itertools
import warnings
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
from torch.distributions.gamma import Gamma
from torch.utils.data import Dataset, DataLoader
from torch.distributions.kl import kl_divergence
from utils import diriKL, normKL, c2d, bad_tensor, warn_tensor, log1mexp#, hyperdist, p_hd
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
                              'R_conc':None,
                              'R_scale':None, 
                              'T':None,
                              'alpha_conc':None,
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
        self.max_cosh = 1e+6
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cpu")
        # Initialize parameters of variational distribution      
        self.params_reset()        
        
        # Initialize the priors fron the priors' dictionary or with 
        # default values
        
        if priors['R_p'] is None:
            R_p = torch.tensor([1., 1.]).to(self.device).to(self.dtype)
        else:
            R_p = priors['R_p'].to(self.device).to(self.dtype) 
            
        if priors['T_p'] is None:
            T_p = torch.ones([2]).to(self.device).to(self.dtype)
        else:
            T_p = priors['T_p'].to(self.device).to(self.dtype)
        
        if priors['alpha_p'] is None:
            alpha_p = torch.tensor([1., 1.]).to(self.device).to(self.dtype)
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
            phis_loc = torch.rand([self.num_nodes,2])
        else:
            phis_loc = self.init_values['phis_loc']
            
        if self.init_values['phis_scale'] is None:
            phis_scale = torch.rand([self.num_nodes])
        else:
            phis_scale = self.init_values['phis_scale']
            
        if self.init_values['R_conc'] is None:
            R_conc = torch.rand()
        else:
            R_conc = self.init_values['R_conc']
            
        if self.init_values['R_scale'] is None:
            R_scale = torch.rand()
        else:
            R_scale = self.init_values['R_scale']
            
        if self.init_values['T'] is None:
            T = torch.rand([2])
        else:
            T = self.init_values['T']
            
        if self.init_values['alpha_conc'] is None:
            alpha_conc = torch.rand()
        else:
            alpha_conc = self.init_values['alpha_conc']
            
        if self.init_values['alpha_scale'] is None:
            alpha_scale = torch.rand()
        else:
            alpha_scale = self.init_values['alpha_scale']
            
        self.rs_loc = torch.nn.Parameter(rs_loc.to(self.device).to(self.dtype))
        self.rs_scale = torch.nn.Parameter(rs_scale.to(self.device).to(self.dtype))
        self.phis_loc = torch.nn.Parameter(phis_loc.to(self.device).to(self.dtype))
        self.phis_scale = torch.nn.Parameter(phis_scale.to(self.device).to(self.dtype))
        self.R_conc = torch.nn.Parameter(R_conc.to(self.device).to(self.dtype))
        self.R_scale = torch.nn.Parameter(R_scale.to(self.device).to(self.dtype))
        self.T = torch.nn.Parameter(T.to(self.device).to(self.dtype))
        self.alpha_conc = torch.nn.Parameter(alpha_conc.to(self.device).to(self.dtype))
        self.alpha_scale = torch.nn.Parameter(alpha_scale.to(self.device).to(self.dtype))
        
        
    def constrained_params(self):
        ''' Return constrained posterior parameters. '''
        return (self.rs_loc,
                torch.exp(self.rs_scale),
                self.phis_loc,
                torch.exp(self.phis_scale),
                torch.exp(self.R_conc),
                torch.exp(self.R_scale),
                torch.exp(self.T),
                torch.exp(self.alpha_conc),
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
                
    def elbo(self, idx1, idx2, weights, debug=False):
        ''' Return evidence lower bound (ELBO) calculated for a nodes batch 
        of size L; also the loss for the training.
        
        ARGUMENTS:
        
        idx1 (torch.int, size: L): start nodes.
        idx2 (torch.int, size: L): finish nodes.
        weights (torch.float, size: L): edges weights.
        
        '''
        
        L = len(weights)   # Batch size
        r_x_loc, r_x_scale, phi_x_loc, phi_x_scale, R_x_conc, R_x_scale, T_x, \
            alpha_x_conc, alpha_x_scale = self.constrained_params() 
        
        edges = torch.where(weights>0, 
                            torch.ones(weights.size()).to(self.device).to(self.dtype), 
                            torch.zeros(weights.size()).to(self.device).to(self.dtype))
        
        R_q = Gamma(R_x_conc, R_x_scale.reciprocal())
        T_q = Beta(T_x[0], T_x[1])
        alpha_q = Gamma(alpha_x_conc, alpha_x_scale.reciprocal())
                    
        loop_count = 0
        while True:
            loop_count += 1
            R_samples = R_q.rsample([self.num_samples]).squeeze(-1).to(self.device).to(self.dtype)
            T_samples = T_q.rsample([self.num_samples]).squeeze(-1).to(self.device).to(self.dtype)
            alpha_samples = alpha_q.rsample([self.num_samples]).squeeze(-1).to(self.device).to(self.dtype)
        
            r_q = Radius(r_x_loc.expand(self.num_samples,self.num_nodes), 
                           r_x_scale.expand(self.num_samples,self.num_nodes), 
                           R_samples.expand(self.num_nodes,self.num_samples).t())
            r_samples = r_q.rsample().to(self.device).to(self.dtype)
        
            alpha_r_i = log1mexp(alpha_samples.expand(L,self.num_samples).t()*r_samples[:,idx1]*2)
            alpha_R = log1mexp(alpha_samples*R_samples)
            alpha_R_ri = - alpha_samples.expand(L,self.num_samples).t()*\
                        (R_samples.expand(L,self.num_samples).t()-r_samples[:,idx1])
            q_ri = r_q.log_prob(r_samples)
            if not (bad_tensor(warn_tensor(alpha_R_ri, 'alpha_R_ri')) \
                or bad_tensor(warn_tensor(alpha_r_i, 'alpha_ri')) \
                or bad_tensor(warn_tensor(alpha_R, 'alpha_R')) \
                or bad_tensor(warn_tensor(q_ri, 'q_ri')) ):
                break
            if loop_count > 100:
                raise Exception('Infinite loop!!!')
        
        phi_q = VonMisesFisher(phi_x_loc, phi_x_scale.unsqueeze(dim=-1))
        phi_samples = phi_q.rsample(self.num_samples).to(self.device).to(self.dtype)#[:,:,:2]
        
        #print(phi_i_samples.shape)
        

#        arcosh_ = lambda x: (torch.clamp(x, min=1.+self.epsilon) + (torch.clamp(x, min=1.+self.epsilon)**2 - 1 ).sqrt()).log()
#        arcosh = lambda x: (x + (x**2 - 1 + self.epsilon).sqrt()).log()
#        hyperdist = lambda rx,ry,fx,fy: arcosh(rx.cosh()*ry.cosh() - rx.sinh()*ry.sinh()*(fx-fy).cos())
#        hd = lambda rx,ry,fx,fy: rx.cosh()*ry.cosh() - rx.sinh()*ry.sinh()*(fx-fy).cos()
        cosh_dist_warn = lambda rx,ry,fx,fy: warn_tensor(rx.cosh()*ry.cosh(), 'coshs') -\
                warn_tensor(rx.sinh()*ry.sinh(), 'sinhs')*(fx-fy).cos()
#        p_hd_ = lambda d,R,T: (1.+((d-R)/(2.*T)).exp()).reciprocal()
#        phd = lambda d,R,T: 0.5 + 0.5*((d-R)/(-4.*T)).tanh()
        p_approx = lambda c,R,T: (1.+(2*c).pow(1./(2.*T))*(-R/(2.*T)).exp()).reciprocal()
        
        def p_app_warn(c,R,T):
            temp1 = (2* warn_tensor(c, 'c')).pow(1./(2.*warn_tensor(T, 'T')))
            temp1_ = torch.where(torch.isnan(temp1), torch.tensor(np.inf).expand(temp1.shape).to(self.dtype), temp1)
#            print(c)
#            print(T)
            temp2 = (- warn_tensor(R, 'R')/(2.* warn_tensor(T, 'T'))).exp()
            return (1. + warn_tensor(temp1_, 'temp1_')*warn_tensor(temp2, 'temp2')).reciprocal()
        
        cosh_dist = cosh_dist_warn(warn_tensor(r_samples[:,idx1], 'r_samples_1'), 
                       warn_tensor(r_samples[:,idx2], 'r_samples_2'), 
                       warn_tensor(c2d(phi_samples[:,idx1]), 'phi_samples_1'), 
                       warn_tensor(c2d(phi_samples[:,idx2]), 'phi_samples_2'))
        cd_clamped = torch.clamp(cosh_dist, min=self.epsilon, max=self.max_cosh)
#        print(cosh_dist)
#        dist = arcosh_(warn_tensor(cosh_dist,'cosh_dist')) #+self.epsilon
        p_raw = p_app_warn(warn_tensor(cd_clamped,'cd_clamped'), R_samples.expand(L,self.num_samples).t(), T_samples.expand(L,self.num_samples).t())
        p_clamped = torch.clamp(warn_tensor(p_raw,'p_raw'), min=self.epsilon, max=1.-self.epsilon)
        prob_edges = Bernoulli(warn_tensor(p_clamped,'p_clamped')).log_prob(edges).mean(dim=0)
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
        elbo1 = - L/self.num_nodes**2 * \
            kl_divergence(R_q, Gamma(self.R_p[0], self.R_p[1].reciprocal())).double()
        if debug: print('-D_kl(R)    >>', str(elbo1))
        #elbo -= 1/self.num_nodes**2 * normKL(alpha_x, self.alpha_p).double()
        elbo2 = - L/self.num_nodes**2 * \
            kl_divergence(alpha_q, Gamma(self.alpha_p[0], self.alpha_p[1].reciprocal())).double()
        if debug: print('-D_kl(alpha)>>', str(elbo2))
        #elbo -= 1/self.num_nodes**2 * diriKL(T_x, self.T_p).double()
        elbo3 = - L/self.num_nodes**2 * \
            kl_divergence(T_q, Beta(self.T_p[0], self.T_p[1])).double()
        if debug: print('-D_kl(T)    >>', str(elbo3)) 
        
        elbo4 = warn_tensor(prob_edges, 'log_pA').sum()
        if debug: print('Prob_edges  >>', str(elbo4))
        elbo5 = L/self.num_nodes * (alpha_R_ri+alpha_r_i).mean(dim=0).sum()
#        print(alpha_R_ri)
#        print(alpha_r_i)
#        elbo5 = L/self.num_nodes * (alpha_R_ri).mean(dim=0).sum() 
        if debug: print('Alpha_R_ri  >>', str(elbo5)) 
        elbo6 = L/self.num_nodes * alpha_samples.log().mean()
        if debug: print('Alpha       >>', str(elbo6)) 
        elbo7 = - L/self.num_nodes * torch.tensor(np.pi*2).log()
#        if debug: print('Log(2*pi)   >>', str(elbo7))
        elbo8 = - L/self.num_nodes * 2 * alpha_R.mean()
        if debug: print('Alpha_R     >>', str(elbo8)) 
#        print(q_ri[:,idx1])
        #print(r_samples)
#        print(r_x_loc, r_x_scale)
#        print(R_samples)
        elbo9 = - L/self.num_nodes * q_ri[:,idx1].mean(dim=0).sum()
        if debug: print('P(q_ri)     >>', str(elbo9))
        q_phi_entropy = phi_q.entropy()[idx1]
#        print(q_phi_entropy)        
        elbo10 = L/self.num_nodes * q_phi_entropy.sum()
        if debug: print('P(q_phii)   >>', str(elbo10))
        elbo = elbo1+elbo2+elbo3+elbo4+elbo5+elbo6+elbo7+elbo8+elbo9+elbo10 
        if debug: print('ELBO >>>>', str(elbo))
        return elbo
        
    def train_(self, epoch_num, debug=False):
        ''' Fit the variational distribution for one epoch.
        
        ARGUMENTS:
        
        epoch_num (int): the epoch's number for printing it out.        
        '''
        t1 = time.time()  # Measure the training time
        total_loss = 0
        for idx1, idx2, data in self.dataloader:
            idx1, idx2, data = idx1.to(self.device), idx2.to(self.device), data.to(self.device)
            loss = - self.elbo(idx1, idx2, data, debug)
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
              lrs=0.1, epochs=10, momentum=None, debug=False):
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
                   curr_loss = self.train_(epoch_counter, debug)
                   epoch_counter +=1
                   self.loss_list.append(curr_loss)
        else:
            # If lrs is a list and epochs is an integer, train the model for 
            # the same number of epochs with each lr
            if isinstance(self.epochs, int):
                for lr in self.lrs:
                    self.optimizer.lr=lr
                    for e in range(self.epochs):
                       curr_loss = self.train_(epoch_counter, debug)
                       epoch_counter +=1
                       self.loss_list.append(curr_loss)
            # If both lrs and epochs are lists, train the model with 
            # corresponding lr and epoch
            else:
                for l in range(len(self.lrs)):
                    self.optimizer.lr=self.lrs[l]
                    for e in range(self.epochs[l]):
                            curr_loss = self.train_(epoch_counter, debug)
                            epoch_counter +=1
                            self.loss_list.append(curr_loss)                 
        self.loss_list = torch.tensor(self.loss_list)
        print('>>>>>>>>>>>> Training is finished.\n')
    
