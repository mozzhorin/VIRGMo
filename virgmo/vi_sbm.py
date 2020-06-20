# -*- coding: utf-8 -*-
"""
Variational inference for stochastic block models.
"""
#import time
import itertools
#import #warnings
#import pickle
#import copy
#import numpy as np
#import matplotlib.pyplot as plt
#import networkx as nx
import torch
from torch.autograd import Variable
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.beta import Beta
#from torch.distributions.normal import Normal
#from torch.utils.data import Dataset, DataLoader
from virgmo.utils import diriKL, gammaKL, normKL, softmax
from virgmo.vi_rg import VI_RG

class VI_SBM(VI_RG):
    '''
    Variational inference for stochastic block models.
    
    PARAMETERS:
    
    num_classes (int): number of classes K
    num_nodes (int): number of nodes N
    etas (torch.Tensor, size: K*N): 
        posterior class assignment probabilities of each node.
    thetas (torch.Tensor, size: K):
        posterior classes probabilities.
    Bs (torch.Tensor, size: K*K*2):
        posterior parameters of Beta probability disrtibutions of edges 
        between nodes of specific classes.
    theta_p (torch.Tensor, size: K):
        prior classes probabilities.
    B_p (torch.Tensor, size: K*K*2):
        prior parameters of Beta probability disrtibutions of edges 
        between nodes of specific classes.
        
    EXAMPLE:
    
    N = 75
    p = torch.tensor([0.2, 0.3, 0.5])
    b = torch.tensor([
        [0.8, 0.1, 0.4],
        [0.1, 0.9, 0.1],
        [0.4, 0.1, 0.8]])
    sbm = SBM(p, b)
    gen_z, gen_A = sbm.generate(N)
    sbm.show(sorted=True)
    dataloader = DataLoader(EdgesDataset(gen_A), 
                            batch_size=10, shuffle=True, num_workers=0)
    vi = VI_SBM(num_nodes=N, num_classes=3)
    vi.train(dataloader, epochs=10)
    vi.summary(gen_A, gen_z)
    
    '''
    def __init__(self, num_classes=2, num_nodes=50, 
                 priors={'theta_p':None, 
                         'B_p':None},
                 init_values={'etas':None, 
                              'thetas':None, 
                              'Bs':None},
                 device=None):
        ''' Initialize VI_SBM model.
        
        ARGUMENTS:
        
        num_classes (int): number of classes K
        num_nodes (int): number of nodes N
        priors (dict of torch.float): priors
        init_values (dict of torch.float): initial values of the variational 
            distribution's parameters.
        ''' 
        super(VI_SBM, self).__init__(num_nodes, priors, 
                                     init_values, device)
        self.num_classes = num_classes
        # Initialize parameters of variational distribution      
        self.params_reset()        
        
        # Initialize the priors fron the priors' dictionary or with 
        # default values
        if priors['theta_p'] is None:
            # Default flat prior for Dirichlet distribution
            theta_p = torch.ones([self.num_classes])
        else:
            theta_p = priors['theta_p']
        if priors['B_p'] is None:
            # Default flat prior for Beta distribution
            B_p = torch.ones([self.num_classes, self.num_classes, 2])
        else:
            B_p = priors['B_p']
            
        self.theta_p = Variable(theta_p, requires_grad=False).to(self.device)
        self.B_p = Variable(B_p, requires_grad=False).to(self.device)
    
    def params_reset(self):
        ''' Reset parameters of the variational distribution from the 
        init_values dictionary or with random values.'''
        
        if self.init_values['etas'] is None:
            etas = torch.rand([self.num_classes, self.num_nodes]).to(self.device)
        elif self.init_values['etas']=='SHORTEST_PATH':
            # Dataloader can be not specified yet
            try:
                etas = self.etas_init().to(self.device)
                print('Initialize etas with shortest path algorithm')
            except:
                etas = torch.rand([self.num_classes, self.num_nodes]).to(self.device)
        else:
            etas = self.init_values['etas'].to(self.device)
            
        if self.init_values['thetas'] is None:
            thetas = torch.rand([self.num_classes]).to(self.device)
        else:
            thetas = self.init_values['thetas'].to(self.device)
            
        if self.init_values['Bs'] is None:
            Bs = torch.rand([self.num_classes, self.num_classes, 2]).to(self.device)
        else:
            Bs = self.init_values['Bs'].to(self.device)
            
        self.etas = torch.nn.Parameter(etas)
        self.thetas = torch.nn.Parameter(thetas)
        self.Bs = torch.nn.Parameter(Bs)
    
    def constrained_params(self):
        ''' Return constrained posterior parameters. '''
        return (softmax(self.etas), 
               torch.exp(self.thetas)+self.epsilon, 
               torch.exp(self.Bs))
    
    def elbo(self, idx1, idx2, weights, debug=False):
        ''' Return evidence lower bound (ELBO) calculated for a nodes batch 
        of size L; also the loss for the training.
        
        ARGUMENTS:
        
        idx1 (torch.int, size: L): start nodes.
        idx2 (torch.int, size: L): finish nodes.
        weights (torch.float, size: L): edges weights.
        
        '''
        
        L = len(weights)   # Batch size
        eta_x, theta_x, B_x = self.constrained_params()   
        
        # Calculate and sum different parts of ELBO
        elbo  = - L / self.num_nodes**2 * diriKL(theta_x, self.theta_p)                
        elbo += - L / self.num_nodes**2 * diriKL(B_x, self.B_p).sum()        
        elbo += 1 / self.num_nodes * self.phi(idx1, eta_x, theta_x) 
        elbo += self.omega(B_x, eta_x, idx1, idx2, weights)        
        return elbo
   
    def qmean(self):
        ''' Return mean values of posterior variational distributions.
        '''
        eta_x, theta_x, B_x = self.constrained_params()
        thetas = Dirichlet(theta_x.data).mean
        Bs = torch.zeros([self.num_classes, self.num_classes])
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                Bs[i,j] = Beta(B_x[i,j,0].data, B_x[i,j,1].data).mean
        return eta_x.data, thetas, Bs
        
    def summary(self, A, z=None):
        ''' Print the summary and plot the sorted adjacency matrix and 
        the loss for one fit.
        
        ARGUMENTS:
        
        A (torch.Tensor, size: N*N): adjacency matrix.
        z (torch.Tensor, size: N*K): binary matrix indicating the true class 
            assignment for each data point.
        '''
        qmean = super(VI_SBM, self).summary(A,z)
        print('Classes probability', qmean[1].numpy())
        print('Edges probability:\n', qmean[2].numpy())
        if len(qmean)>3:
            return qmean
        
    def class_accuracy(self, z, eta=None):
        ''' Return the best accuracy of nodes' class assignments for all 
        permutations of class' labels. 
        
        ARGUMENTS:
        
        z (torch.Tensor, size: N*K): binary matrix indicating the true class 
            assignment for each data point.
        eta (torch.Tensor, size: K*N): 
            posterior class assignment probabilities of each node.
        '''
        if eta is None:
            eta = self.qmean()[0]
        eta = eta.cpu()
        z = z.cpu()
        pred = eta.argmax(dim=0)
        truth = z.argmax(dim=-1).float()
        perm_list = list(itertools.permutations(range(self.num_classes)))
        pred_modify = pred + self.num_classes
        # Calculate all permutations predicted/true class names
        perms = torch.empty(len(perm_list), len(truth))        
        for p in range(len(perm_list)):
            tmp = pred_modify.clone()
            for i in range(self.num_classes):
                tmp = torch.where(tmp==(self.num_classes+i), 
                                  torch.tensor(perm_list[p][i]), tmp)
            perms[p] = tmp.clone()
        compare = torch.empty(len(perm_list), len(truth))
        for p in range(len(perm_list)):
            compare[p] = perms[p]==truth
        # Choose the permutation with the highest accuracy rate
        return compare.sum(dim=-1).div(len(truth)).max()

###############################################################################
        
class VI_DCSBM(VI_SBM):
    '''
    Variational inference for degree-corrected stochastic block models.
    
    Parameters:
    
    num_classes (int): number of classes K
    num_nodes (int): number of nodes N
    etas (torch.Tensor, size: K*N): 
        posterior class assignment probabilities of each node.
    thetas (torch.Tensor, size: K):
        posterior classes probabilities.
    Bs (torch.Tensor, size: K*K*2):
        posterior parameters of Beta probability disrtibutions corresponding to
        probabilities of edges between nodes of specific classes.
    deltas (torch.Tensor, size: N*2):
        posterior parameters of Normal probability disrtibutions corresponding 
        to expected degree of each node.
    theta_p (torch.Tensor, size: K):
        prior classes probabilities.
    B_p (torch.Tensor, size: K*K*2):
        prior parameters of Beta probability disrtibutions corresponding to
        probabilities of edges between nodes of specific classes.
    delta_p (torch.Tensor, size: N*2):
        prior parameters of Normal probability disrtibutions corresponding to
        expected degree of each node.
        
    EXAMPLE:
    
    N = 75
    p = torch.tensor([0.2, 0.3, 0.5])
    b = torch.tensor([
            [0.8, 0.1, 0.4],
            [0.1, 0.9, 0.1],
            [0.4, 0.1, 0.8]])
    delta = torch.tensor([[0.,1.], [2.,1.], [-2.,4.]])
    dcsbm = DCSBM(p, b, delta)
    z, A = dcsbm.generate(N)
    dcsbm.show(sorted=True)
    dataloader = DataLoader(EdgesDataset(A), 
                            batch_size=10, shuffle=True, num_workers=0)
    vi = VI_DCSBM(num_nodes=N, num_classes=3)
    vi.train(dataloader, epochs=10)
    vi.summary(A,z)
    
    '''
    def __init__(self, num_classes=2, num_nodes=50, num_samples=10,
                 priors={'theta_p':None, 
                         'B_p':None, 
                         'delta_p':None},
                 init_values={'etas':None, 
                              'thetas':None, 
                              'Bs':None, 
                              'deltas':None},
                 device=None):
        ''' Initialize VI_DCSBM model.
        
        ARGUMENTS:
        
        num_classes (int): number of classes K
        num_nodes (int): number of nodes N
        num_samples (int): number of samples to calculated the expectation of
            ELBO's parts, when it cannot be done analyticaly.
        priors (dict of torch.float): priors
        init_values (dict of torch.float): initial values of the variational 
            distribution's parameters.
        '''                           
        super(VI_DCSBM, self).__init__(num_classes, num_nodes, priors, 
                                       init_values, device)
        self.num_samples = num_samples

        if priors['delta_p'] is None:
            # Default standard normal prior
            delta_p = torch.ones([self.num_nodes, 2])*torch.tensor([0.,1])
        else:
            delta_p = priors['delta_p']
        self.delta_p = Variable(delta_p, requires_grad=False).to(self.device)
        
    def params_reset(self):
        ''' Reset parameters of the variational distribution from the 
        init_values dictionary or with random values.
        '''
        super(VI_DCSBM, self).params_reset()
        
        if self.init_values['deltas'] is None:
            deltas = torch.rand([self.num_nodes, 2]).to(self.device)
        else:
            deltas = self.init_values['deltas'].to(self.device)            
         
        self.deltas = torch.nn.Parameter(deltas)
    
    def constrained_params(self):
        ''' Returned constrained posterior parameters.'''
        return (softmax(self.etas), torch.exp(self.thetas),
                torch.exp(self.Bs), self.deltas)     
    
    def elbo(self, idx1, idx2, weights, debug=False):
        ''' Return evidence lower bound (ELBO) calculated for a nodes batch 
        of size L; also the loss for the training.
        
        Arguments:
        
        idx1 (torch.int, size: L): start nodes.
        idx2 (torch.int, size: L): finish nodes.
        weights (torch.float, size: L): edges weights.
        
        '''        
        L = len(weights)   # Batch size        
        eta_x, theta_x, B_x, delta_x = self.constrained_params()          
        elbo  = - L / self.num_nodes**2 * diriKL(theta_x, self.theta_p)                
        elbo += - L / self.num_nodes**2 * diriKL(B_x, self.B_p).sum()        
        elbo +=   1 / self.num_nodes * self.phi(idx1, eta_x, theta_x)
        elbo += - 1 / self.num_nodes * normKL(delta_x, self.delta_p).sum()
        elbo += self.omega_approx(B_x, eta_x, delta_x, idx1, idx2, weights)
        return elbo
   
    def qmean(self):
        ''' Return mean values of posterior variational distributions.
        '''
        eta_x, theta_x, B_x, delta_x = self.constrained_params()
        thetas = Dirichlet(theta_x.data).mean
        Bs = torch.zeros([self.num_classes, self.num_classes])
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                Bs[i,j] = Beta(B_x[i,j,0].data, B_x[i,j,1].data).mean
        deltas = delta_x[:,0].data
        return eta_x.data, thetas, Bs, deltas
    
        
    def summary(self, A, z=None):
        ''' Print the summary and plot the sorted adjacency matrix and 
        the loss for one fit.
        
        ARGUMENTS:
        
        A (torch.Tensor, size: N*N): adjacency matrix.
        z (torch.Tensor, size: N*K): binary matrix indicating the true class 
            assignment for each data point.
        '''
        qmean = super(VI_DCSBM, self).summary(A,z)
        print('Expected degree:\n', qmean[3].numpy())
        if len(qmean)>4:
            return qmean
        
###############################################################################
        
class VI_WDCSBM(VI_DCSBM):
    '''
    Variational inference for weighted degree-corrected stochastic block models.
    
    Parameters:
    
    num_classes (int): number of classes K
    num_nodes (int): number of nodes N
    etas (torch.Tensor, size: K*N): 
        posterior class assignment probabilities of each node.
    thetas (torch.Tensor, size: K):
        posterior classes probabilities.
    Bs (torch.Tensor, size: K*K*2):
        posterior parameters of Beta probability disrtibutions corresponding to
        probabilities of edges between nodes of specific classes.
    deltas (torch.Tensor, size: N*2):
        posterior parameters of Normal probability disrtibutions corresponding 
        to expected degree of each node.
    mus (torch.Tensor, size: K*K*2):
        posterior parameters of Normal probability disrtibutions corresponding 
        to mean of weights
    taus (torch.Tensor, size: K*K*2):
        posterior parameters of Gamma probability disrtibutions corresponding 
        to precision of weights
    theta_p (torch.Tensor, size: K):
        prior classes probabilities.
    B_p (torch.Tensor, size: K*K*2):
        prior parameters of Beta probability disrtibutions corresponding to
        probabilities of edges between nodes of specific classes.
    delta_p (torch.Tensor, size: N*2):
        prior parameters of Normal probability disrtibutions corresponding to
        expected degree of each node.
    mu_p (torch.Tensor, size: K*K*2):
        prior parameters of Normal probability disrtibutions corresponding 
        to mean of weights
    tau_p (torch.Tensor, size: K*K*2):
        prior parameters of Gamma probability disrtibutions corresponding 
        to precision of weights
        
    EXAMPLE:
    
    N = 75
    p = torch.tensor([0.2, 0.3, 0.5])
    b = torch.tensor([
            [0.8, 0.1, 0.4],
            [0.1, 0.9, 0.1],
            [0.4, 0.1, 0.8]])
    delta = torch.tensor([[0.,1.], [2.,1.], [-2.,4.]])
    g_mu = torch.tensor([
            [10., 5., 2.],
            [5., 10., 2.],
            [2., 2., 20.]])
    g_tau = torch.ones([3,3])*2
    model = WDCSBM(p, b, delta, g_mu.log(), g_tau) 
    z, A = model.generate(N, directed=True)
    model.show(sorted=True)       
    dataloader = DataLoader(EdgesDataset(A), 
                            batch_size=25, shuffle=True, num_workers=0)
    vi = VI_WDCSBM(num_nodes=N, num_classes=3)
    vi.train(dataloader, epochs=10)
    vi.summary(A, z)
    
    '''
    def __init__(self, num_classes=2, num_nodes=50, num_samples=10,
                 priors={'theta_p':None, 
                         'B_p':None, 
                         'delta_p':None,
                         'mu_p':None,
                         'tau_p':None},
                 init_values={'etas':None, 
                              'thetas':None, 
                              'Bs':None, 
                              'deltas':None,
                              'mus':None,
                              'taus':None},
                 device=None):
        ''' Initialize VI_WDCSBM model.
        
        ARGUMENTS:
        
        num_classes (int): number of classes K
        num_nodes (int): number of nodes N
        num_samples (int): number of samples to calculated the expectation of
            ELBO's parts, when it cannot be done analyticaly.
        priors (dict of torch.float): priors
        init_values (dict of torch.float): initial values of the variational 
            distribution's parameters.
        '''                           
        super(VI_WDCSBM, self).__init__(num_classes, num_nodes, num_samples, 
                                        priors, init_values, device) 
        if priors['mu_p'] is None:
            # Default normal prior
            mu_p = torch.ones([self.num_classes, self.num_classes, 2])        
        else:
            mu_p = priors['mu_p']
            
        if priors['tau_p'] is None:
            # Default normal prior
            tau_p = torch.ones([self.num_classes, self.num_classes, 2])
        else:
            tau_p = priors['tau_p']

        self.mu_p = Variable(mu_p, requires_grad=False).to(self.device)
        self.tau_p = Variable(tau_p, requires_grad=False).to(self.device)
        
    def params_reset(self):
        ''' Reset parameters of the variational distribution from the 
        init_values dictionary or with random values.
        '''
        super(VI_WDCSBM, self).params_reset()
            
        if self.init_values['mus'] is None:
            mus = torch.rand([self.num_classes, self.num_classes, 2]).to(self.device)
        else:
            mus = self.init_values['mus'].to(self.device)
            
        if self.init_values['taus'] is None:
            taus = torch.rand([self.num_classes, self.num_classes, 2]).to(self.device)
        else:
            taus = self.init_values['taus'].to(self.device)

        self.mus = torch.nn.Parameter(mus)
        self.taus = torch.nn.Parameter(taus)
    
    def constrained_params(self):
        ''' Returned constrained posterior parameters.'''
        return (softmax(self.etas), torch.exp(self.thetas)+self.epsilon,
                torch.exp(self.Bs), self.deltas, 
                torch.exp(self.mus), torch.exp(self.taus))
    
    def elbo(self, idx1, idx2, weights, debug=False):
        ''' Return evidence lower bound (ELBO) calculated for a nodes batch 
        of size L; also the loss for the training.
        
        Arguments:
        
        idx1 (torch.int, size: L): start nodes.
        idx2 (torch.int, size: L): finish nodes.
        data_values (torch.float, size: L): edges weights.
        
        '''        
        L = len(weights)   # Batch size
        eta_x, theta_x, B_x, delta_x, mu_x, tau_x = self.constrained_params()
        elbo  = - L / self.num_nodes**2 * diriKL(theta_x, self.theta_p)                
        elbo += - L / self.num_nodes**2 * diriKL(B_x, self.B_p).sum()        
        elbo +=   1 / self.num_nodes * self.phi(idx1, eta_x, theta_x)
        elbo += - 1 / self.num_nodes * normKL(delta_x, self.delta_p).sum()
        elbo += - L / self.num_nodes**2 * normKL(mu_x, self.mu_p).sum()
        elbo += - L / self.num_nodes**2 * gammaKL(tau_x, self.tau_p).sum()
        elbo += self.omega_approx(B_x, eta_x, delta_x, idx1, idx2, weights)
        elbo += self.psi(eta_x, mu_x, tau_x, idx1, idx2, weights)
        return elbo

    def qmean(self):
        ''' Return mean values of posterior variational distributions.
        '''
        eta_x, theta_x, B_x, delta_x, mu_x, tau_x = self.constrained_params()
        thetas = Dirichlet(theta_x.data).mean
        Bs = torch.zeros([self.num_classes, self.num_classes])
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                Bs[i,j] = Beta(B_x[i,j,0].data, B_x[i,j,1].data).mean
        deltas = delta_x[:,0].data
        mus = mu_x[:,:,0].data
        taus = tau_x.data.prod(dim=-1)
        return eta_x.data, thetas, Bs, deltas, mus, taus
           
    def summary(self, A, z=None):
        ''' Print the summary and plot the sorted adjacency matrix and 
        the loss for one fit.
        
        ARGUMENTS:
        
        A (torch.Tensor, size: N*N): adjacency matrix.
        z (torch.Tensor, size: N*K): binary matrix indicating the true class 
            assignment for each data point.
        '''
        qmean = super(VI_WDCSBM, self).summary(A,z)
        print('Expected mean weight:\n', qmean[4].numpy())
        print('Expected precision weight:\n', qmean[5].numpy())
        if len(qmean)>6:
            return qmean
        
        
###############################################################################
        
class VI_WSBM(VI_SBM):
    '''
    Variational inference for weighted stochastic block models.
    
    Parameters:
    
    num_classes (int): number of classes K
    num_nodes (int): number of nodes N
    etas (torch.Tensor, size: K*N): 
        posterior class assignment probabilities of each node.
    thetas (torch.Tensor, size: K):
        posterior classes probabilities.
    Bs (torch.Tensor, size: K*K*2):
        posterior parameters of Beta probability disrtibutions corresponding to
        probabilities of edges between nodes of specific classes.
    mus (torch.Tensor, size: K*K*2):
        posterior parameters of Normal probability disrtibutions corresponding 
        to mean of weights
    taus (torch.Tensor, size: K*K*2):
        posterior parameters of Gamma probability disrtibutions corresponding 
        to precision of weights
    theta_p (torch.Tensor, size: K):
        prior classes probabilities.
    B_p (torch.Tensor, size: K*K*2):
        prior parameters of Beta probability disrtibutions corresponding to
        probabilities of edges between nodes of specific classes.
    mu_p (torch.Tensor, size: K*K*2):
        prior parameters of Normal probability disrtibutions corresponding 
        to mean of weights
    tau_p (torch.Tensor, size: K*K*2):
        prior parameters of Gamma probability disrtibutions corresponding 
        to precision of weights
        
    EXAMPLE:
    
    N = 75
    p = torch.tensor([0.2, 0.3, 0.5])
    b = torch.tensor([
            [0.8, 0.1, 0.4],
            [0.1, 0.9, 0.1],
            [0.4, 0.1, 0.8]])
    delta = torch.tensor([[100.,1.], [100.,1.], [100.,1.]])
    g_mu = torch.tensor([
            [10., 5., 2.],
            [5., 10., 2.],
            [2., 2., 20.]])
    g_tau = torch.ones([3,3])*2
    model = WDCSBM(p, b, delta, g_mu.log(), g_tau) 
    z, A = model.generate(N, directed=True)
    model.show(sorted=True)       
    dataloader = DataLoader(EdgesDataset(A), 
                            batch_size=25, shuffle=True, num_workers=0)
    vi = VI_WSBM(num_nodes=N, num_classes=3)
    vi.train(dataloader, epochs=10)
    vi.summary(A, z)
    
    '''
    def __init__(self, num_classes=2, num_nodes=50,
                 priors={'theta_p':None, 
                         'B_p':None, 
                         'mu_p':None,
                         'tau_p':None},
                 init_values={'etas':None, 
                              'thetas':None, 
                              'Bs':None,
                              'mus':None,
                              'taus':None},
                 device=None):
        ''' Initialize VI_WSBM model.
        
        ARGUMENTS:
        
        num_classes (int): number of classes K
        num_nodes (int): number of nodes N
        num_samples (int): number of samples to calculated the expectation of
            ELBO's parts, when it cannot be done analyticaly.
        priors (dict of torch.float): priors
        init_values (dict of torch.float): initial values of the variational 
            distribution's parameters.
        '''                           
        super(VI_WSBM, self).__init__(num_classes, num_nodes, 
                                      priors, init_values, device) 
        if priors['mu_p'] is None:
            # Default normal prior
            mu_p = torch.ones([self.num_classes, self.num_classes, 2])        
        else:
            mu_p = priors['mu_p']
            
        if priors['tau_p'] is None:
            # Default normal prior
            tau_p = torch.ones([self.num_classes, self.num_classes, 2])
        else:
            tau_p = priors['tau_p']

        self.mu_p = Variable(mu_p, requires_grad=False).to(self.device)
        self.tau_p = Variable(tau_p, requires_grad=False).to(self.device)
        
    def params_reset(self):
        ''' Reset parameters of the variational distribution from the 
        init_values dictionary or with random values.
        '''
        super(VI_WSBM, self).params_reset()
            
        if self.init_values['mus'] is None:
            mus = torch.rand([self.num_classes, self.num_classes, 2]).to(self.device)
        else:
            mus = self.init_values['mus'].to(self.device)
            
        if self.init_values['taus'] is None:
            taus = torch.rand([self.num_classes, self.num_classes, 2]).to(self.device)
        else:
            taus = self.init_values['taus'].to(self.device)

        self.mus = torch.nn.Parameter(mus)
        self.taus = torch.nn.Parameter(taus)
    
    def constrained_params(self):
        ''' Returned constrained posterior parameters.'''
        return (softmax(self.etas), torch.exp(self.thetas)+self.epsilon,
                torch.exp(self.Bs), torch.exp(self.mus), torch.exp(self.taus))
    
    def elbo(self, idx1, idx2, weights, debug=False):
        ''' Return evidence lower bound (ELBO) calculated for a nodes batch 
        of size L; also the loss for the training.
        
        Arguments:
        
        idx1 (torch.int, size: L): start nodes.
        idx2 (torch.int, size: L): finish nodes.
        data_values (torch.float, size: L): edges weights.
        
        '''        
        L = len(weights)   # Batch size
        eta_x, theta_x, B_x, mu_x, tau_x = self.constrained_params()
        elbo  = - L / self.num_nodes**2 * diriKL(theta_x, self.theta_p)                
        elbo += - L / self.num_nodes**2 * diriKL(B_x, self.B_p).sum()        
        elbo +=   1 / self.num_nodes * self.phi(idx1, eta_x, theta_x)
        elbo += - L / self.num_nodes**2 * normKL(mu_x, self.mu_p).sum()
        elbo += - L / self.num_nodes**2 * gammaKL(tau_x, self.tau_p).sum()
        elbo += self.omega(B_x, eta_x, idx1, idx2, weights)
        elbo += self.psi(eta_x, mu_x, tau_x, idx1, idx2, weights)
        return elbo

    def qmean(self):
        ''' Return mean values of posterior variational distributions.
        '''
        eta_x, theta_x, B_x, mu_x, tau_x = self.constrained_params()
        thetas = Dirichlet(theta_x.data).mean
        Bs = torch.zeros([self.num_classes, self.num_classes])
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                Bs[i,j] = Beta(B_x[i,j,0].data, B_x[i,j,1].data).mean
        mus = mu_x[:,:,0].data
        taus = tau_x.data.prod(dim=-1)
        return eta_x.data, thetas, Bs, mus, taus
           
    def summary(self, A, z=None):
        ''' Print the summary and plot the sorted adjacency matrix and 
        the loss for one fit.
        
        ARGUMENTS:
        
        A (torch.Tensor, size: N*N): adjacency matrix.
        z (torch.Tensor, size: N*K): binary matrix indicating the true class 
            assignment for each data point.
        '''
        qmean = super(VI_WSBM, self).summary(A,z)
        print('Expected mean weight:\n', qmean[3].numpy())
        print('Expected precision weight:\n', qmean[4].numpy())
        if len(qmean)>5:
            return qmean

        
###############################################################################
        
class VI_WCRG(VI_RG):
    '''
    Variational inference for weighted stochastic block models for complete graphs.
    
    Parameters:
    
    num_classes (int): number of classes K
    num_nodes (int): number of nodes N
    etas (torch.Tensor, size: K*N): 
        posterior class assignment probabilities of each node.
    thetas (torch.Tensor, size: K):
        posterior classes probabilities.
    Bs (torch.Tensor, size: K*K*2):
        posterior parameters of Beta probability disrtibutions corresponding to
        probabilities of edges between nodes of specific classes.
    mus (torch.Tensor, size: K*K*2):
        posterior parameters of Normal probability disrtibutions corresponding 
        to mean of weights
    taus (torch.Tensor, size: K*K*2):
        posterior parameters of Gamma probability disrtibutions corresponding 
        to precision of weights
    theta_p (torch.Tensor, size: K):
        prior classes probabilities.
    B_p (torch.Tensor, size: K*K*2):
        prior parameters of Beta probability disrtibutions corresponding to
        probabilities of edges between nodes of specific classes.
    mu_p (torch.Tensor, size: K*K*2):
        prior parameters of Normal probability disrtibutions corresponding 
        to mean of weights
    tau_p (torch.Tensor, size: K*K*2):
        prior parameters of Gamma probability disrtibutions corresponding 
        to precision of weights
        
    EXAMPLE:
    
    N = 75
    p = torch.tensor([0.2, 0.3, 0.5])
    g_mu = torch.tensor([
            [10., 5., 2.],
            [5., 50., 2.],
            [2., 2., 20.]])
    g_tau = torch.ones([3,3])*2
    model = WCRG(p, g_mu.log(), g_tau) 
    z, A = model.generate(N, directed=True)
    model.show(sorted=True)   
    dataloader = DataLoader(EdgesDataset(A), 
                            batch_size=25, shuffle=True, num_workers=0)
    vi = VI_WCRG(num_nodes=N, num_classes=3)
    vi.train(dataloader, epochs=10)
    vi.summary(A, z)
    
    '''
    def __init__(self, num_classes=2, num_nodes=50,
                 priors={'theta_p':None, 
                         'mu_p':None,
                         'tau_p':None},
                 init_values={'etas':None, 
                              'thetas':None, 
                              'mus':None,
                              'taus':None},
                 device=None):
        ''' Initialize VI_WCRG model.
        
        ARGUMENTS:
        
        num_classes (int): number of classes K
        num_nodes (int): number of nodes N
        num_samples (int): number of samples to calculated the expectation of
            ELBO's parts, when it cannot be done analyticaly.
        priors (dict of torch.float): priors
        init_values (dict of torch.float): initial values of the variational 
            distribution's parameters.
        '''                           
        super(VI_WCRG, self).__init__(num_nodes, priors, init_values, device) 
        self.num_classes = num_classes
        self.params_reset()                               
        if priors['theta_p'] is None:
            # Default flat prior for Dirichlet distribution
            theta_p = torch.ones([self.num_classes])
        else:
            theta_p = priors['theta_p']
        if priors['mu_p'] is None:
            # Default normal prior
            mu_p = torch.ones([self.num_classes, self.num_classes, 2])        
        else:
            mu_p = priors['mu_p']
            
        if priors['tau_p'] is None:
            # Default normal prior
            tau_p = torch.ones([self.num_classes, self.num_classes, 2])
        else:
            tau_p = priors['tau_p']
        self.theta_p = Variable(theta_p, requires_grad=False).to(self.device)
        self.mu_p = Variable(mu_p, requires_grad=False).to(self.device)
        self.tau_p = Variable(tau_p, requires_grad=False).to(self.device)
        
    def params_reset(self):
        ''' Reset parameters of the variational distribution from the 
        init_values dictionary or with random values.
        '''
        if self.init_values['etas'] is None:
            etas = torch.rand([self.num_classes, self.num_nodes]).to(self.device)
        elif self.init_values['etas']=='SHORTEST_PATH':
            # Dataloader can be not specified yet
            try:
                etas = self.etas_init().to(self.device)
                print('Initialize etas with shortest path algorithm')
            except:
                etas = torch.rand([self.num_classes, self.num_nodes]).to(self.device)
        else:
            etas = self.init_values['etas'].to(self.device)
            
        if self.init_values['thetas'] is None:
            thetas = torch.rand([self.num_classes]).to(self.device)
        else:
            thetas = self.init_values['thetas'].to(self.device)
            
        if self.init_values['mus'] is None:
            mus = torch.rand([self.num_classes, self.num_classes, 2]).to(self.device)
        else:
            mus = self.init_values['mus'].to(self.device)
            
        if self.init_values['taus'] is None:
            taus = torch.rand([self.num_classes, self.num_classes, 2]).to(self.device)
        else:
            taus = self.init_values['taus'].to(self.device)
            
        self.etas = torch.nn.Parameter(etas)
        self.thetas = torch.nn.Parameter(thetas)
        self.mus = torch.nn.Parameter(mus)
        self.taus = torch.nn.Parameter(taus)
    
    def constrained_params(self):
        ''' Returned constrained posterior parameters.'''
        return (softmax(self.etas), torch.exp(self.thetas)+self.epsilon,
                torch.exp(self.mus), torch.exp(self.taus))
    
    def elbo(self, idx1, idx2, weights, debug=False):
        ''' Return evidence lower bound (ELBO) calculated for a nodes batch 
        of size L; also the loss for the training.
        
        Arguments:
        
        idx1 (torch.int, size: L): start nodes.
        idx2 (torch.int, size: L): finish nodes.
        data_values (torch.float, size: L): edges weights.
        
        '''        
        L = len(weights)   # Batch size
        eta_x, theta_x, mu_x, tau_x = self.constrained_params()
        elbo  = - L / self.num_nodes**2 * diriKL(theta_x, self.theta_p)
        if debug: print('D_KL(theta)>>', str(elbo))
        elbo += - L / self.num_nodes**2 * normKL(mu_x, self.mu_p).sum()
        if debug: print('+D_KL(mu)  >>', str(elbo))
        elbo += - L / self.num_nodes**2 * gammaKL(tau_x, self.tau_p).sum()
        if debug: print('+D_KL(tau) >>', str(elbo))
        elbo +=   1 / self.num_nodes * self.phi(idx1, eta_x, theta_x)
        if debug: print('+Phi       >>', str(elbo))        
        elbo += self.psi(eta_x, mu_x, tau_x, idx1, idx2, weights)
        if debug: print('+Psi       >>', str(elbo))
        return elbo

    def qmean(self):
        ''' Return mean values of posterior variational distributions.
        '''
        eta_x, theta_x, mu_x, tau_x = self.constrained_params()
        thetas = Dirichlet(theta_x.data).mean
        mus = mu_x[:,:,0].data
        taus = tau_x.data.prod(dim=-1)
        return eta_x.data, thetas, mus, taus
           
    def summary(self, A, z=None):
        ''' Print the summary and plot the sorted adjacency matrix and 
        the loss for one fit.
        
        ARGUMENTS:
        
        A (torch.Tensor, size: N*N): adjacency matrix.
        z (torch.Tensor, size: N*K): binary matrix indicating the true class 
            assignment for each data point.
        '''
        qmean = super(VI_WCRG, self).summary(A,z)
        print('Classes probability', qmean[1].numpy())
        print('Expected mean weight:\n', qmean[2].numpy())
        print('Expected precision weight:\n', qmean[3].numpy())
        if len(qmean)>4:
            return qmean

