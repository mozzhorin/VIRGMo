# -*- coding: utf-8 -*-
"""
Variational inference for random graph models.
"""
import time
#import itertools
#import #warnings
#import pickle
import copy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
#from torch.autograd import Variabl#e
#from torch.distributions.dirichlet import Dirichlet
from torch.distributions.beta import Beta
from torch.distributions.normal import Normal
#from torch.utils.data import Dataset, DataLoader
#from virgmo.utils import diriKL, gammaKL, normKL, softmax

class VI_RG(torch.nn.Module):
    '''
    Variational inference for random graph models.
    
    PARAMETERS:
    
    num_classes (int): number of classes K
    num_nodes (int): number of nodes N
    
    '''
    def __init__(self, num_nodes=50, priors=None, 
                 init_values=None, device=None):
        ''' Initialize the model.
        
        ARGUMENTS:
        
        num_classes (int): number of classes K
        num_nodes (int): number of nodes N
        priors (dict of torch.float): priors
        init_values (dict of torch.float): initial values of the variational 
            distribution's parameters.
        '''
        super(VI_RG, self).__init__()        
        self.num_nodes = num_nodes
        self.init_values = init_values
        self.epsilon = 1e-14  # Keeps some params away from extream values
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cpu")
    
    def params_reset(self):
        ''' Reset the parameters of the variational distribution.'''
        pass
    
    def etas_init(self, centers=None):
        ''' Initialize etas with shortest path algorithm.
        '''
        A_weights = self.dataloader.dataset.get_matrix()
        A_edges = torch.where(A_weights!=0, 
                              torch.ones(A_weights.size()), 
                              torch.zeros(A_weights.size()))
        G = nx.Graph(A_edges.numpy())
        dist = np.zeros([self.num_classes,self.num_nodes])
        if centers is None:
            centers = np.random.randint(0,self.num_nodes,self.num_classes)
        for c in range(len(centers)):
            dist_ = np.ones(self.num_nodes)*np.inf
            sp = nx.single_source_shortest_path_length(G, centers[c])
            dist_[list(sp.keys())] = list(sp.values())
            dist[c] = dist_
        etas = 1/(dist+1)
        return torch.tensor(etas).float()
    
    def constrained_params(self):
        ''' Return constrained posterior parameters. '''
        pass
    
    def elbo(self, idx1, idx2, weights, debug=False):
        ''' Return the evidence lower bound (ELBO) calculated for a nodes 
        batch of size L; also the loss for the training.
        
        ARGUMENTS:
        
        idx1 (torch.int, size: L): start nodes.
        idx2 (torch.int, size: L): finish nodes.
        weights (torch.float, size: L): edges' weights.
        debug (bool): debug mode 
        
        '''        
        pass
    
    def phi(self, idx1, eta_x, theta_x):
        ''' Returns part of ELBO (see the documentation).
        
        ARGUMENTS:
    
        idx1 (torch.int, size: L): start nodes.
        eta_x (torch.Tensor, size: K*L): constrained etas' batch.
        theta_x (torch.Tensor, size: K): constrained thetas.
        '''
        temp = (theta_x.digamma()-theta_x.sum().digamma()).unsqueeze(0).transpose(0,1)
        return torch.mul(eta_x[:,idx1],temp-eta_x[:,idx1].log()).sum() 
        
    def omega(self, B_x, eta_x, idx1, idx2, weights):
        ''' Returns part of ELBO (see the documentation). 
        Calculated analytycally.
        
        ARGUMENTS:
    
        idx1 (torch.int, size: L): start nodes.
        idx2 (torch.int, size: L): finish nodes.
        data (torch.float, size: L): edges weights.
        '''
        # Transform weighted edges to unweighted (0/1 weights)
        edges = torch.where(weights>0, 
                            torch.ones(weights.size()).to(self.device), 
                            torch.zeros(weights.size()).to(self.device))
        c = torch.where(edges==1, 
                        torch.zeros(edges.size(), dtype=torch.long).to(self.device), 
                        torch.ones(edges.size(), dtype=torch.long).to(self.device))
        log_pA = B_x[:,:,c].digamma()-B_x.sum(dim=2).digamma().unsqueeze(-1)
        temp = torch.mul(eta_x[:,idx1].unsqueeze(0),
                         eta_x[:,idx2].unsqueeze(1)).transpose(0,1)
        return torch.mul(log_pA, temp).sum()
        
    def omega_approx(self, B_x, eta_x, delta_x, idx1, idx2, weights):
        ''' Returns part of ELBO (see the documentation).
        Estimated through sampling.
        
        ARGUMENTS:        
        
        B_x (torch.Tensor, size: K*K*2): constrained Bs.
        eta_x (torch.Tensor, size: K*L): constrained etas' batch.
        delta_x (torch.Tensor, size: N*2): constrained deltas.
        idx1 (torch.int, size: L): start nodes.
        idx2 (torch.int, size: L): finish nodes.            
        weights (torch.float, size: L): edges' weights.
        '''
        delta_i = Normal(delta_x[idx1,0],delta_x[idx1,1])\
                    .rsample([self.num_samples]).to(self.device)
        delta_j = Normal(delta_x[idx2,0],delta_x[idx2,1])\
                    .rsample([self.num_samples]).to(self.device)     
        B_ij = Beta(B_x[:,:,0],B_x[:,:,1])\
                    .rsample([self.num_samples]).to(self.device)
        sig = torch.sigmoid(delta_i+delta_j)
        B = sig.unsqueeze(-1).unsqueeze(-1) * B_ij.unsqueeze(1)
        ElogB = B.log().mean(dim=0)
        ElogB_ = (1. - B).log().mean(dim=0)
        edges = torch.where(weights>0, 
                            torch.ones(weights.size()).to(self.device), 
                            torch.zeros(weights.size()).to(self.device))
        log_pA = edges.unsqueeze(-1).unsqueeze(-1)*ElogB \
               + (1.-edges).unsqueeze(-1).unsqueeze(-1)*ElogB_            
        temp = torch.mul(eta_x[:,idx1].unsqueeze(0),
                         eta_x[:,idx2].unsqueeze(1)).transpose(0,2)
        return torch.mul(log_pA, temp).sum()

    def psi(self, eta_x, mu_x, tau_x, idx1, idx2, weights):
        ''' Returns part of ELBO (see the documentation). 
        Calculated analytycally, used on weighted graphs.
        
        ARGUMENTS:
    
        eta_x (torch.Tensor, size: K*L): constrained etas' batch.
        mu_x (torch.Tensor, size: K*K*2): constrained mus' batch.
        tau_x (torch.Tensor, size: K*K*2): constrained taus' batch.
        idx1 (torch.int, size: L): start nodes.
        idx2 (torch.int, size: L): finish nodes.            
        weights (torch.float, size: L): edges' weights.
        '''
        weights_ = weights[weights.nonzero()].squeeze(-1)
        idx1_ = idx1[weights.nonzero()].squeeze(-1)
        idx2_ = idx2[weights.nonzero()].squeeze(-1)            
        log_g = weights_.log().unsqueeze(-1).unsqueeze(-1)
        temp1 = - 0.5*torch.tensor(np.pi*2).to(self.device).log()\
            + 0.5*(tau_x[:,:,0].digamma()+tau_x[:,:,1].log()).unsqueeze(0)\
            - log_g - 0.5*tau_x.prod(dim=-1)\
                         *(log_g.pow(2)-2*log_g*mu_x[:,:,0].unsqueeze(0)\
                           + mu_x.pow(2).sum(dim=-1).unsqueeze(0))                               
        temp2 = torch.mul(eta_x[:,idx1_].unsqueeze(0), eta_x[:,idx2_].unsqueeze(1))
        return torch.mul(temp1, temp2.transpose(0,2)).sum()
   
    def qmean(self):
        ''' Return mean values of posterior variational distributions.
        '''
        return()
    
    def train_(self, epoch_num, debug=False, verbose=True):
        ''' Fit the variational distribution for one epoch.
        
        ARGUMENTS:
        
        epoch_num (int): the epoch's number for printing it out.        
        '''
        t1 = time.time()  # Measure the training time
        total_loss = 0
        for idx1, idx2, data in self.dataloader:
            idx1, idx2, data = idx1.to(self.device), idx2.to(self.device), data.to(self.device)
            loss = - self.elbo(idx1, idx2, data, debug=debug)
            self.optimizer.zero_grad()
            loss.backward(retain_graph=False)
            self.optimizer.step()
            total_loss += loss.detach().clone()
#            if data.sum()<=0:
#                warnings.warn('Batch is empty! Your graph is to sparse, increase the batch size!')            
        t2 = time.time()
        tl = total_loss.cpu().data.numpy().item()
        if verbose:
            print('Epoch %d | LR: %.2f | Total loss: %.2f | Epoch time %.2f'\
                  % (epoch_num+1, self.optimizer.lr, tl, (t2-t1)))
        return tl
    
    def train(self, dataloader, optimizer=torch.optim.RMSprop, 
              lrs=0.1, epochs=10, momentum=None, debug=False, verbose=True):
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
        self.optimizer = optimizer(self.parameters()) 
        
        # Set the momentum if specified
        if momentum is not None:
            self.optimizer.momentum = momentum
        
        # Total loss for traning monitoring
        self.loss_list = [] 
        
        # Training loop
        if verbose:
            print('>>>>>>>>>>>> Start training...')
        epoch_counter = 0
        # Check if lrs is a float, then run only one loop
        if isinstance(self.lrs, float) or isinstance(self.lrs, int):
            self.optimizer.lr=self.lrs
            for e in range(self.epochs):
                   curr_loss = self.train_(epoch_counter, debug, verbose)
                   epoch_counter +=1
                   self.loss_list.append(curr_loss)
        else:
            # If lrs is a list and epochs is an integer, train the model for 
            # the same number of epochs with each lr
            if isinstance(self.epochs, int):
                for lr in self.lrs:
                    self.optimizer.lr=lr
                    for e in range(self.epochs):
                       curr_loss = self.train_(epoch_counter, debug, verbose)
                       epoch_counter +=1
                       self.loss_list.append(curr_loss)
            # If both lrs and epochs are lists, train the model with 
            # corresponding lr and epoch
            else:
                for l in range(len(self.lrs)):
                    self.optimizer.lr=self.lrs[l]
                    for e in range(self.epochs[l]):
                            curr_loss = self.train_(epoch_counter, debug, verbose)
                            epoch_counter +=1
                            self.loss_list.append(curr_loss)                 
        self.loss_list = torch.tensor(self.loss_list)
        if verbose:
            print('>>>>>>>>>>>> Training is finished.\n')
        
    def multi_train(self, dataloader, optimizer=torch.optim.RMSprop, momentum=None,
                    lrs=0.1, epochs=10, trials=10, init_states=None, debug=False, verbose=False):
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
        num_param = len(self.qmean())
        if not num_param:
            self.multi_results = None
            self.state_dicts = None
        else:
            results = [[]]
            losses, state_dicts = [], []
            for i in range(num_param-1):
                results.append([])
            print('>>>>>>> Start multi-training...')
            for i in range(trials):
                try:
                    t1 = time.time()  # Measure the training time
                    if init_states is None:
                        self.params_reset()
                    else:
                        self.load_state_dict(init_states[i])
                    if verbose:
                        print('>>>>>>> Training trial #%d \n' % (i+1))
                    self.train(dataloader, optimizer=optimizer, epochs=epochs, lrs=lrs,
                               momentum=momentum, debug=debug, verbose=verbose)
                    means = self.qmean()
                    for j in range(num_param):
                        results[j].append(means[j])
                    losses.append(self.loss_list)
                    state_dicts.append(self.state_dict())
                    t2 = time.time()
                    if not verbose:
                        print('>>> Trial %d/%d | Final loss: %.2f | Trial time %.2f'\
                               % (i+1, trials, self.loss_list[-1], (t2-t1)))
                except Exception as e: 
                    print(e)
            for i in range(num_param):
                results[i]=torch.stack(results[i])
            results.append(torch.stack(losses))
            self.multi_results = results
            self.state_dicts = state_dicts
            
    def get_multi_losses(self):
        return self.multi_results[-1]
    
    def get_multi_means(self):
        return self.multi_results[:-1]
            
    def pyramid_train(self, dataloader, optimizer=torch.optim.RMSprop, momentum=None,
                    lrs=[0.1, 0.05, 0.01], 
                    epochs=[10,10,10],  
                    trials=[100,10,1]):
        ''' Fit the model several times in several 'layers':
                #layers = len(trials) = len(epochs) = len(lrs)
        At each layer l only m=trials[l] fits with the smallest loss from 
        the previous layer are trained for e=epochs[l] with the learning rate
        r=lrs[l].
        For example, if trials=[100,10,1], at first, 100 models are trained,
        then only 10 best of them, and then only the best model is trained. 
        Used when the posterior has many local optimas and the quality of fit 
        highly depends on the initial values of the variational distribution's 
        parameters.
        
        ARGUMENTS:
        
        dataloader (torch.utils.data.DataLoader):
            dataloader to iterate over the list of edges.
        optimizer (str, {rmsprop, adagrad, adam, asgd, sgd}): 
            optimizer's name
        momentum (float): momentum factor.
        lrs (list of floats, size; l): learning rates for each layer.            
        epochs (list of int, size; l): number of training epochs for each layer.
        trials (list of int, size; l): number of training trials.        
         
        '''
        num_layers = len(trials)
        for l in range(num_layers):
            if not l:
                best_dicts=None
            else:
                last_losses = self.multi_results[-1][:,-1]
                best_runs = last_losses.argsort()[:trials[l]]
                print('Best runs:', best_runs)
                print('with losses:', last_losses.sort().values[:trials[l]])
                best_dicts = []
                for b in best_runs:
                    best_dicts.append(copy.deepcopy(self.state_dicts[b.item()]))
            self.multi_train(dataloader, optimizer=optimizer, momentum=momentum,
                             lrs=lrs[l], epochs=epochs[l], trials=trials[l],
                             init_states=best_dicts) 
        
        
    def summary(self, A, z=None):
        ''' Plots the sorted adjacency matrix and prints the summary and 
        the loss for one fit.
        
        ARGUMENTS:
        
        A (torch.Tensor, size: N*N): adjacency matrix.
        z (torch.Tensor, size: N*K): binary matrix indicating the true class 
            assignment for each data point.
        '''
        qmean = self.qmean()
        if len(qmean)>0:
            plt.title('Adjacency matrix')
            plt.imshow(A[:,qmean[0].argmax(dim=0).argsort()]\
                        [qmean[0].argmax(dim=0).argsort(),:])
            plt.show()            
            plt.plot(self.loss_list.numpy())
            plt.title('Training loss (ELBO)')
            plt.ylabel('Loss')
            plt.xlabel('Epochs')
            plt.show() 
        if not z is None:
            print('Latent class accuracy:', self.class_accuracy(z).numpy().item())
        if len(qmean)>1:
            return qmean