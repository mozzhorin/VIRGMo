# -*- coding: utf-8 -*-
"""
Variational inference for hyperbilic random graph models.
"""
#import time
#import itertools
#import warnings
#import pickle
#import copy
#import networkx as nx
import numpy as np
#import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
#from torch.distributions.bernoulli import Bernoulli
from torch.distributions.beta import Beta
from torch.distributions.normal import Normal
#from torch.distributions.log_normal import LogNormal
from torch.distributions.gamma import Gamma
#from torch.utils.data import Dataset, DataLoader
from torch.distributions.kl import kl_divergence
from virgmo.utils import (unit_circle, log1mexp, cosh_dist, c2d, bad_tensor, 
     warn_tensor, clmpd_log1pexp, clmpd_log1pexp_)
#from graph_models import EdgesDataset, EdgesIterableDataset, HRG
from virgmo.vi_rg import VI_RG
from virgmo.distributions.von_mises_fisher import VonMisesFisher
from virgmo.distributions.radius import Radius

softmax = torch.nn.Softmax(dim=0)

class VI_HRG(VI_RG):
    '''
    Variational inference for hyperbilic random graph models.
    
    PARAMETERS:    
    
    num_nodes (int): number of nodes N
    
    rs_loc (torch.Tensor, size: N): posterior location of r-coordinate of each node.
    rs_scale (torch.Tensor, size: N): posterior scale of r-coordinate of each node.
    phis_loc (torch.Tensor, size: N*2): posterior location of phi-coordinate of each node.
    phis_scale (torch.Tensor, size: N): posterior scale of phi-coordinate of each node.
    R_conc (torch.Tensor, size: 1): posterior concentration of R.
    R_scale (torch.Tensor, size: 1): posterior scale of R.
    alpha_conc (torch.Tensor, size: 1): posterior concentration of alpha.
    alpha_scale (torch.Tensor, size: 1): posterior scale of alpha.
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
    def __init__(self, num_nodes=50, num_samples=20, dtype=torch.double,
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
                 fixed={'R':None, 
                        'T':None,
                        'alpha':None},
                clamp_dist=False,
                 device=None):
        ''' Initialize the model.
        
        ARGUMENTS:
        
        num_nodes (int): number of nodes N
        num_samples (int): number of samples
        priors (dict of torch.float): priors
        init_values (dict of torch.float): initial values of the variational 
            distribution's parameters.
        fixed (dict of torch.float): fixed parameters
        '''
        super(VI_HRG, self).__init__(num_nodes, priors, init_values, device)        
#        self.num_nodes = num_nodes
#        self.init_values = init_values
        self.num_samples = num_samples

        self.fixed = fixed
        self.dtype = dtype
        self.epsilon = 1e-5  # Keeps some params away from extream values
        self.max_cosh = 1e+10
        self.clamp_dist = clamp_dist
#        if device is not None:
#            self.device = device
#        else:
#            self.device = torch.device("cpu")
        
        # Fix the parameters if it's required
        self.Rf, self.Tf, self.alphaf = fixed['R'], fixed['T'], fixed['alpha']  
        
        # Initialize parameters of variational distribution      
        self.params_reset()        
        
        # Initialize the priors fron the priors' dictionary or with 
        # default values
        
        if priors['R_p'] is None:
            R_p = torch.tensor([10., 1.]).to(self.device).to(self.dtype)
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
            rs_loc = torch.rand([self.num_nodes]).log()
        else:
            rs_loc = self.init_values['rs_loc'].clone()
            
        if self.init_values['rs_scale'] is None:
            rs_scale = (torch.ones([self.num_nodes])/4).log()
        else:
            rs_scale = self.init_values['rs_scale'].clone()
            
        if self.init_values['phis_loc'] is None:
            phis_loc = Normal(0,1).sample([self.num_nodes,2])
        else:
            phis_loc = self.init_values['phis_loc'].clone()
            
        if self.init_values['phis_scale'] is None:
            phis_scale = (torch.ones([self.num_nodes])*20).log()
        else:
            phis_scale = self.init_values['phis_scale'].clone()
            
        if self.init_values['R_conc'] is None:
            R_conc = torch.tensor(20.).log()
        else:
            R_conc = self.init_values['R_conc'].clone()
            
        if self.init_values['R_scale'] is None:
            R_scale = torch.tensor(.4).log()
        else:
            R_scale = self.init_values['R_scale'].clone()
            
        if self.init_values['T'] is None:
            T = torch.tensor([1., 15.]).log()
        else:
            T = self.init_values['T'].clone()
            
        if self.init_values['alpha_conc'] is None:
            alpha_conc = torch.tensor(75.).log()
        else:
            alpha_conc = self.init_values['alpha_conc'].clone()
            
        if self.init_values['alpha_scale'] is None:
            alpha_scale = torch.tensor(0.01).log()
        else:
            alpha_scale = self.init_values['alpha_scale'].clone()
            
        self.rs_loc = torch.nn.Parameter(rs_loc.to(self.device).to(self.dtype))
        self.rs_scale = torch.nn.Parameter(rs_scale.to(self.device).to(self.dtype))
        self.phis_loc = torch.nn.Parameter(phis_loc.to(self.device).to(self.dtype))
        self.phis_scale = torch.nn.Parameter(phis_scale.to(self.device).to(self.dtype))
        if self.Rf is None:
            self.R_conc = torch.nn.Parameter(R_conc.to(self.device).to(self.dtype))
            self.R_scale = torch.nn.Parameter(R_scale.to(self.device).to(self.dtype))
        else:
            self.R_conc, self.R_scale = torch.tensor(0.), torch.tensor(0.)
        if self.Tf is None:
            self.T = torch.nn.Parameter(T.to(self.device).to(self.dtype))
        else:
            self.T = torch.tensor([0.,0.])
        if self.alphaf is None:
            self.alpha_conc = torch.nn.Parameter(alpha_conc.to(self.device).to(self.dtype))
            self.alpha_scale = torch.nn.Parameter(alpha_scale.to(self.device).to(self.dtype))
        else:
            self.alpha_conc, self.alpha_scale = torch.tensor(0.), torch.tensor(0.)
        
        
        
    def constrained_params(self):
        ''' Return constrained posterior parameters. '''
        return (self.rs_loc,
                torch.exp(self.rs_scale),
                unit_circle(self.phis_loc),
                torch.exp(self.phis_scale),
                torch.exp(self.R_conc),
                torch.exp(self.R_scale),
                torch.exp(self.T),
                torch.exp(self.alpha_conc),
                torch.exp(self.alpha_scale))
                

                
    def elbo(self, idx1, idx2, weights, debug=False, likelihood=False):
        ''' Return evidence lower bound (ELBO, training loss) or likelihood 
        for a nodes batch of size L.
        
        ARGUMENTS:
        
        idx1 (torch.int, size: L): start nodes.
        idx2 (torch.int, size: L): finish nodes.
        weights (torch.float, size: L): edges weights.
        likelihood (bool): return the batch's likelihood instead of ELBO?
        debug (bool): print intermediate values for debugging?
        
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
        
        # Because of numerical instability we put the sampling of several parameters
        # into a loop, wich can be exited only if the samping produces 'stable' results             
        loop_count = 0
        while True:
            loop_count += 1
            if self.Rf is None:
                R_samples = R_q.rsample([self.num_samples]).squeeze(-1).to(self.device).to(self.dtype)
            else:
                R_samples = torch.tensor(self.Rf).clone().detach().to(self.device).to(self.dtype).expand([self.num_samples])
            if self.Tf is None:
                T_samples = T_q.rsample([self.num_samples]).squeeze(-1).to(self.device).to(self.dtype)
            else:
                T_samples = torch.tensor(self.Tf).clone().detach().to(self.device).to(self.dtype).expand([self.num_samples])
            if self.alphaf is None:
                alpha_samples = alpha_q.rsample([self.num_samples]).squeeze(-1).to(self.device).to(self.dtype)
            else:
                alpha_samples = torch.tensor(self.alphaf).clone().detach().to(self.device).to(self.dtype).expand([self.num_samples])
        
            r_q = Radius(r_x_loc.expand(self.num_samples,self.num_nodes), 
                           r_x_scale.expand(self.num_samples,self.num_nodes), 
                           R_samples.expand(self.num_nodes,self.num_samples).t())
            r_samples = r_q.rsample().to(self.device).to(self.dtype)
        
            l1e_a_ri = log1mexp(alpha_samples.expand(L,self.num_samples).t()*r_samples[:,idx1]*2)
            l1e_a_R = log1mexp(alpha_samples*R_samples)
            a_R_ri = - alpha_samples.expand(L,self.num_samples).t()*\
                        (R_samples.expand(L,self.num_samples).t()-r_samples[:,idx1])
            r_q_lp = r_q.log_prob(r_samples)
            r_q_lp = torch.clamp(r_q_lp, min=-1e+4)
            phi_q = VonMisesFisher(phi_x_loc, phi_x_scale.unsqueeze(dim=-1))
            phi_samples = phi_q.rsample(self.num_samples).to(self.device).to(self.dtype)
            cd_raw = cosh_dist(r_samples[:,idx1], r_samples[:,idx2], 
                           c2d(phi_samples[:,idx1]), c2d(phi_samples[:,idx2]))
            #if debug:
                #print(cd_raw)
            if self.clamp_dist:
                cd_raw = torch.clamp(cd_raw, max=self.max_cosh)
            edes_prob_arg = ((cd_raw*2).log()-R_samples.expand(L,self.num_samples).t())/(2*T_samples.expand(L,self.num_samples).t())
            if not (bad_tensor(warn_tensor(a_R_ri, 'a_R_ri')) \
                    or bad_tensor(warn_tensor(l1e_a_ri, 'l1e_a_ri')) \
                    or bad_tensor(warn_tensor(l1e_a_R, 'l1e_a_R')) \
                    or bad_tensor(warn_tensor(r_q_lp, 'r_q_lp')) \
                    or bad_tensor(warn_tensor(edes_prob_arg, 'edes_prob_arg')) ):
                break
            if loop_count > 200:
                raise Exception('Sampling runs in an infinite loop!!!')
        
        elbo1 = 0 if not self.Rf is None else - L/self.num_nodes**2 * \
            kl_divergence(R_q, Gamma(self.R_p[0], self.R_p[1].reciprocal())).to(self.dtype)
        if debug: print('-D_kl(R)    >>', str(elbo1))
        
        elbo2 = 0 if not self.alphaf is None else - L/self.num_nodes**2 * \
            kl_divergence(alpha_q, Gamma(self.alpha_p[0], self.alpha_p[1].reciprocal())).to(self.dtype)
        if debug: print('-D_kl(alpha)>>', str(elbo2))
       
        elbo3 = 0 if not self.Tf is None else - L/self.num_nodes**2 * \
            kl_divergence(T_q, Beta(self.T_p[0], self.T_p[1])).to(self.dtype)
        if debug: print('-D_kl(T)    >>', str(elbo3)) 
        
        lp = edges*clmpd_log1pexp(edes_prob_arg) + (1-edges)*clmpd_log1pexp_(edes_prob_arg)
        elbo4 = lp.mean(dim=0).sum()
        if debug: print('Prob_edges  >>', str(elbo4))
        
        elbo5 = 1/self.num_nodes * (a_R_ri+l1e_a_ri).mean(dim=0).sum()
        if debug: print('a_R_ri  >>', str(elbo5)) 
        
        elbo6 = L/self.num_nodes**2 * alpha_samples.log().mean()
        if debug: print('Alpha       >>', str(elbo6)) 
        
        elbo7 = - L/self.num_nodes**2 * torch.tensor(np.pi*2).log()
#        if debug: print('Log(2*pi)   >>', str(elbo7))
        
        elbo8 = - L/self.num_nodes**2 * 2 * l1e_a_R.mean()
        if debug: print('l1e_a_R     >>', str(elbo8)) 

        elbo9 = - 1/self.num_nodes * r_q_lp[:,idx1].mean(dim=0).sum()
        if debug: print('P(q_r)     >>', str(elbo9))
        
        q_phi_entropy = phi_q.entropy()[idx1]
        elbo10 = 1/self.num_nodes * q_phi_entropy.sum()
        if debug: print('P(q_phii)   >>', str(elbo10))
        
        elbo = elbo1+elbo2+elbo3+elbo4+elbo5+elbo6+elbo7+elbo8+elbo9+elbo10 
        lh = elbo4+elbo5+elbo6+elbo7+elbo8
        if debug: print('ELBO >>>>', str(elbo))
        if debug: print('Batch likelihood >>>>', str(lh))
        if likelihood: 
            return lh
        else:
            return elbo
    
    def likelihood(self, debug=False):
        ''' Compute the model likelihood.'''        
        total_lh = 0
        for idx1, idx2, data in self.dataloader:
            idx1, idx2, data = idx1.to(self.device), idx2.to(self.device), data.to(self.device)
            loss = self.elbo(idx1, idx2, data, debug=debug, likelihood=True)
            total_lh += loss.detach().clone()            
        return total_lh.item()
        
    def elbo_full(self, debug=False):
        ''' Compute the full elbo without training steps. '''        
        elbo = 0
        for idx1, idx2, data in self.dataloader:
            idx1, idx2, data = idx1.to(self.device), idx2.to(self.device), data.to(self.device)
            loss = self.elbo(idx1, idx2, data, debug=debug)
            elbo += loss.detach().clone()            
        return elbo.item()
        
    def qmean(self):
        ''' Return mean values of posterior variational distributions.'''
        r_x_loc, r_x_scale, phi_x_loc, phi_x_scale, R_x_conc, R_x_scale, T_x, \
            alpha_x_conc, alpha_x_scale = self.constrained_params()
        if self.Rf is None:    
            R = Gamma(R_x_conc, R_x_scale.reciprocal()).mean.detach()
        else:
            R = torch.tensor(self.Rf).to(self.device).to(self.dtype)
        if self.alphaf is None:
            alpha = Gamma(alpha_x_conc, alpha_x_scale.reciprocal()).mean.detach()
        else:
            alpha = torch.tensor(self.alphaf).to(self.device).to(self.dtype)
        if self.Tf is None:
            T = Beta(T_x[0], T_x[1]).mean.detach()
        else:
            T = torch.tensor(self.Tf).to(self.device).to(self.dtype)
        rs = Radius(r_x_loc, r_x_scale, R.expand([self.num_nodes])).mean.detach()
        phis = VonMisesFisher(phi_x_loc, phi_x_scale.unsqueeze(dim=-1)).mean.detach()
        return rs, phis, R, T, alpha
    
    def posterior_samples(self, num_samples=20):
        ''' Return the posterior samples of hyperbolic coordinates, 
        used for drawing the node positions.'''
        
        r_x_loc, r_x_scale, phi_x_loc, phi_x_scale, R_x_conc, R_x_scale, T_x, \
            alpha_x_conc, alpha_x_scale = self.constrained_params()
            
        if self.Rf is None:    
            R = Gamma(R_x_conc, R_x_scale.reciprocal()).mean.detach()
        else:
            R = torch.tensor(self.Rf).to(self.device).to(self.dtype)
        rs = Radius(r_x_loc, r_x_scale, R.expand([self.num_nodes])).sample([num_samples])
        phis = VonMisesFisher(phi_x_loc, phi_x_scale.unsqueeze(dim=-1)).sample(num_samples)
        return torch.stack((rs,c2d(phis)), dim=-1)
    
    
    def edge_lh(self, idx1, idx2, weights):
        ''' Calculate the likelihood of a single edge.
        
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
        
        # Because of numerical instability we put the sampling of several parameters
        # into a loop, wich can be exited only if the samping produces 'stable' results             
        loop_count = 0
        while True:
            loop_count += 1
            if self.Rf is None:
                R_samples = R_q.rsample([self.num_samples]).squeeze(-1).to(self.device).to(self.dtype)
            else:
                R_samples = torch.tensor(self.Rf).clone().detach().to(self.device).to(self.dtype).expand([self.num_samples])
            if self.Tf is None:
                T_samples = T_q.rsample([self.num_samples]).squeeze(-1).to(self.device).to(self.dtype)
            else:
                T_samples = torch.tensor(self.Tf).clone().detach().to(self.device).to(self.dtype).expand([self.num_samples])
            if self.alphaf is None:
                alpha_samples = alpha_q.rsample([self.num_samples]).squeeze(-1).to(self.device).to(self.dtype)
            else:
                alpha_samples = torch.tensor(self.alphaf).clone().detach().to(self.device).to(self.dtype).expand([self.num_samples])
        
            r_q = Radius(r_x_loc.expand(self.num_samples,self.num_nodes), 
                           r_x_scale.expand(self.num_samples,self.num_nodes), 
                           R_samples.expand(self.num_nodes,self.num_samples).t())
            r_samples = r_q.rsample().to(self.device).to(self.dtype)
        
            l1e_a_ri = log1mexp(alpha_samples.expand(L,self.num_samples).t()*r_samples[:,idx1]*2)
            l1e_a_R = log1mexp(alpha_samples*R_samples)
            a_R_ri = - alpha_samples.expand(L,self.num_samples).t()*\
                        (R_samples.expand(L,self.num_samples).t()-r_samples[:,idx1])
            r_q_lp = r_q.log_prob(r_samples)
            r_q_lp = torch.clamp(r_q_lp, min=-1e+4)
            phi_q = VonMisesFisher(phi_x_loc, phi_x_scale.unsqueeze(dim=-1))
            phi_samples = phi_q.rsample(self.num_samples).to(self.device).to(self.dtype)
            cd_raw = cosh_dist(r_samples[:,idx1], r_samples[:,idx2], 
                           c2d(phi_samples[:,idx1]), c2d(phi_samples[:,idx2]))
            edes_prob_arg = ((cd_raw*2).log()-R_samples.expand(L,self.num_samples).t())/(2*T_samples.expand(L,self.num_samples).t())
            if not (bad_tensor(warn_tensor(a_R_ri, 'a_R_ri')) \
                    or bad_tensor(warn_tensor(l1e_a_ri, 'l1e_a_ri')) \
                    or bad_tensor(warn_tensor(l1e_a_R, 'l1e_a_R')) \
                    or bad_tensor(warn_tensor(r_q_lp, 'r_q_lp')) \
                    or bad_tensor(warn_tensor(edes_prob_arg, 'edes_prob_arg')) ):
                break
            if loop_count > 200:
                raise Exception('Sampling runs in an infinite loop!!!')
        
            
        lp = edges*clmpd_log1pexp(edes_prob_arg) + (1-edges)*clmpd_log1pexp_(edes_prob_arg)
        return lp.mean(dim=0)
    
    def get_A_lh(self, dataloader):
        ''' Compute the likelihood of A (without Dataloader).'''        
        self.dataloader = dataloader
        A_lh = torch.zeros([self.num_nodes, self.num_nodes]).to(self.device).to(self.dtype)
        for idx1, idx2, data in self.dataloader:
            idx1, idx2, data = idx1.to(self.device), idx2.to(self.device), data.to(self.device)
            lh = self.edge_lh(idx1, idx2, data)
            A_lh[idx1, idx2] = lh.detach().clone()            
        return A_lh