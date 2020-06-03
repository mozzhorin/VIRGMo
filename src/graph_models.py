# -*- coding: utf-8 -*-
"""
Random graph models.
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions.multinomial import Multinomial
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.poisson import Poisson
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal
from torch.distributions.log_normal import LogNormal
from torch.distributions.uniform import Uniform
from torch.utils.data import Dataset, DataLoader, IterableDataset
from utils import *

class Graphon():
    ''' Graphon model.
    
    Parameters:
    
    u (torch.Tensor, size: number of nodes): 
        uniform distributed random variable
    W (function with 2 arguments): 
        graphon
    A (torch.Tensor, size: number of nodes * number of nodes):
        adjacency matrix.
        
    Example:

    N = 75
    G = Graphon()
    u,A = G.generate(N)
    G.show(sorted=True)
    '''
    
    def __init__(self, W=None):
        if W is None:        
            self.W = lambda u,v: torch.mm(u.unsqueeze(1), v.unsqueeze(0))
        else:
            self.W = W
        self.z = None
        self.A = None
            
    def generate(self, n, directed=True):
        ''' Generate the adjacency matrix A and the nodes attribute u.'''
        self.u = Uniform(0,1).sample([n])
        self.A = self.sample_A(self.W(self.u, self.u), directed)
        return self.u, self.A
    
    def sample_A(self, W, directed=True):
        ''' Sample the adjacence matrix A from Bernoulli distribution
        with probabilities W.'''
        A = Bernoulli(W).sample()        
        if not directed:
            A = A.triu(diagonal=1) + A.triu(diagonal=1).t()
        else:
            A = (1. - torch.eye(A.shape[0]))*A        
        return(A)
        
#    def get_W(self):
#        ''' Returns the edges probability matrix.'''
#        return self.W(self.u, self.u)
        
    def show_W(self, u=None, size=(6,6), cmap='viridis'):
        ''' Plot the edges probability matrix.'''
        if u is None:
            u = self.u
        w = self.W(u, u)
        plt.figure(figsize=size)
        plt.imshow(w, cmap=cmap)
        plt.colorbar()
        plt.show()
        
    def show(self, u=None, A=None, sorted=False, weights=True,
             size=(6,6), cmap='viridis'):
        ''' Plot the adjacency matrix; if sorted: accordingly to u. '''
        if u is None:
            u = self.u
        if A is None:
            A = self.A
        if not weights:
            A = torch.where(A>0, 
                            torch.ones(A.size()), 
                            torch.zeros(A.size()))
        plt.figure(figsize=size)
        if sorted:
            order = u.argsort()
            plt.imshow(A[:,order][order,:], cmap=cmap)        
        else:
            plt.imshow(A, cmap=cmap)
        plt.colorbar()
        plt.show()
        
        
class WeightedGraphon(Graphon):
    ''' Weifgted graphon. Weights have a Log-Normal distribution.
    
    Parameters:
    
    u (torch.Tensor, size: number of nodes): 
        uniform distributed random variable
    W (function with 2 arguments): 
        graphon
    mu (function with 2 arguments): 
        function of means of weights distributions.
    tau (function with 2 arguments): 
        function of precision of weights distributions. Variance is 1/tau.
    A (torch.Tensor, size: number of nodes * number of nodes):
        adjacency matrix.
        
    Example:

    N = 75
    G = WeightedGraphon()
    u,A = G.generate(N)
    G.show(sorted=True)
    
    '''
    
    def __init__(self, W=None, mu=None, tau=None):
        if W is None:        
            self.W = lambda u,v: torch.mm(u.unsqueeze(1), v.unsqueeze(0))
        else:
            self.W = W
        if mu is None:        
            self.mu = lambda u,v: (torch.mm(u.unsqueeze(1), v.unsqueeze(0))*20).log()
        else:
            self.mu = mu
        if tau is None:        
            self.tau = lambda u,v: torch.ones(u.size())*5.
        else:
            self.tau = tau
        self.u = None
        self.A = None
            
    def generate(self, n, directed=True):
        self.u = Uniform(0,1).sample([n])
        A = self.sample_A(self.W(self.u, self.u), directed)
        X = LogNormal(self.mu(self.u, self.u), 1./self.tau(self.u,self.u)\
                      .pow(0.5)).sample()
        self.A = A*X
        return self.u, self.A
    
###############################################################################

class SBM(Graphon):
    ''' Stochastic block model.
    
    Parameters:
    
    theta (torch.Tensor, size: number of classes):
        classes probabilities.
    B (torch.Tensor, size: number of classes * number of classes): 
        edges probabilities between nodes of specific classes.
    z (torch.Tensor, size: number of classes * number of nodes): 
        nodes assignment to a specific class.
    A (torch.Tensor, size: number of nodes * number of nodes):
        adjacency matrix.
        
    Example:

    N = 75
    p = torch.tensor([0.2, 0.3, 0.5])
    b = torch.tensor([
            [0.8, 0.1, 0.4],
            [0.1, 0.9, 0.1],
            [0.4, 0.1, 0.8]])
    model = SBM(p, b)
    z, A = model.generate(N, directed=True)
    model.show(sorted=True)        
        
    '''
    
    def __init__(self, theta, B):
        self.theta = theta 
        self.B = B
        self.z = None
        self.A = None
        
    def generate(self, n, directed=True):
        ''' Generate the adjacency matrix A and the nodes assignments z.'''
        self.z = Multinomial(1, self.theta).sample(sample_shape=[n])
        A_prob = torch.mm(torch.mm(self.z,self.B), self.z.t())
        self.A = self.sample_A(A_prob, directed)
        return(self.z, self.A)
        
        
    def show(self, z=None, A=None, sorted=False, weights=True,
             size=(6,6), cmap='viridis'):
        ''' Plot the adjacency matrix; if sorted: accordingly to 
        the assinged class. '''
        if z is None:
            z = self.z
        if A is None:
            A = self.A
        if not weights:
            A = torch.where(A>0, 
                            torch.ones(A.size()), 
                            torch.zeros(A.size()))
        plt.figure(figsize=size)
        if sorted:
            order = z.argmax(dim=1).argsort()
            plt.imshow(A[:,order][order,:], cmap=cmap)        
        else:
            plt.imshow(A, cmap=cmap)
        plt.colorbar()
        plt.show()
        
    def show_W(self):
        pass

class DCSBM(SBM):
    ''' Degree-corected stochastic block model.
    
    Parameters:
    
    theta (torch.Tensor, size: number of classes):
        classes probabilities.
    B (torch.Tensor, size: number of classes * number of classes): 
        edges probabilities between nodes of specific classes.
    z (torch.Tensor, size: number of classes * number of nodes): 
        nodes assignment to a specific class.
    A (torch.Tensor, size: number of nodes * number of nodes):
        adjacency matrix.
    delta_distr (torch.Tensor, size: number of classes * 2):
        [mean, scale] of Normal distribution of the expected degree of nodes
        of each class. Means lay approx. in (-6,6).
        
    Example:

    N = 75
    p = torch.tensor([0.2, 0.3, 0.5])
    b = torch.tensor([
            [0.8, 0.1, 0.4],
            [0.1, 0.9, 0.1],
            [0.4, 0.1, 0.8]])
    delta = torch.tensor([[0.,1.], [2.,1.], [-2.,4.]])
    model = DCSBM(p, b, delta)
    z, A = model.generate(N, directed=True)
    model.show(sorted=True)        
        
    '''
    
    def __init__(self, theta, B, delta_distr):
        self.theta = theta 
        self.B = B
        self.delta_distr = delta_distr
        self.z = None
        self.A = None
        
    def generate(self, n, directed=True):
        ''' Generate the adjacence matrix A and nodes assignments z.'''
        self.z = Multinomial(1, self.theta).sample(sample_shape=[n])
        delta_classes = Normal(self.delta_distr.t()[0], self.delta_distr.t()[1]).sample(sample_shape=[n])
        self.delta = (delta_classes*self.z).sum(dim=1)
        A_prob = torch.mm(torch.mm(self.z,self.B), self.z.t())\
            * torch.sigmoid(self.delta.unsqueeze(-1) + self.delta.unsqueeze(-1).t())
        self.A = self.sample_A(A_prob, directed)
        return(self.z, self.A)

class WSBM(SBM):
    ''' Weighted stochastic block model.
    
    Parameters:
    
    theta (torch.Tensor, size: number of classes):
        classes probabilities.
    B (torch.Tensor, size: number of classes * number of classes): 
        edges probabilities between nodes of specific classes.
    z (torch.Tensor, size: number of classes * number of nodes): 
        nodes assignment to a specific class.
    A (torch.Tensor, size: number of nodes * number of nodes):
        weighted adjacency matrix.
    g_mu (torch.Tensor, size: number of classes * number of classes): 
        mean weights of edges between nodes of specific classes.
    g_tau (torch.Tensor, size: number of classes * number of classes): 
        precision weights of edges between nodes of specific classes.
        
    Example:

    N = 75
    p = torch.tensor([0.2, 0.3, 0.5])
    b = torch.tensor([
            [0.8, 0.1, 0.4],
            [0.1, 0.9, 0.1],
            [0.4, 0.1, 0.8]])
    g_mu = torch.tensor([
            [10., 5., 2.],
            [5., 10., 2.],
            [2., 2., 20.]])
    g_tau = torch.ones([3,3])*2
    model = WSBM(p, b, g_mu.log(), g_tau) 
    z, A = model.generate(N, directed=True)
    model.show(sorted=True)       
        
    '''
    
    def __init__(self, theta, B, g_mu, g_tau):
        self.theta = theta 
        self.B = B
        self.g_mu = g_mu
        self.g_tau = g_tau
        self.z = None
        self.A = None
        
    def generate(self, n, directed=True, weight_interval=None):
        ''' Generate the adjacence matrix A and nodes assignments z.'''
        self.z = Multinomial(1, self.theta).sample(sample_shape=[n])
        A_prob = torch.mm(torch.mm(self.z,self.B), self.z.t())
        x = self.sample_A(A_prob, directed)  
        weights_loc = torch.mm(torch.mm(self.z,self.g_mu), self.z.t())
        weights_scale = 1/torch.mm(torch.mm(self.z,self.g_tau), self.z.t()).pow(0.5)
        w = LogNormal(weights_loc,weights_scale).sample()
        if not weight_interval is None:
            w = torch.where(w < weight_interval[0], 
                            torch.ones(w.size())*weight_interval[0], 
                            w)
            w = torch.where(w > weight_interval[1], 
                            torch.ones(w.size())*weight_interval[1], 
                            w)
        if not directed:
            w = w.triu(diagonal=1) + w.triu(diagonal=1).t()
        self.A = x*w       
        return(self.z, self.A)
    
    def show(self, z=None, A=None, sorted=False, weights=True, 
             size=(6,6), cmap='viridis', zero_edges_nan=False):
        ''' Plot the adjacency matrix; if sorted: accordingly to 
        the assinged class. '''
        if z is None:
            z = self.z.clone()
        if A is None:
            A = self.A.clone()
        if not weights:
            A = torch.where(A>0, 
                            torch.ones(A.size()), 
                            torch.zeros(A.size()))
            
        if zero_edges_nan:
            A[A==0] = np.nan
        plt.figure(figsize=size)
        if sorted:
            order = z.argmax(dim=1).argsort()
            plt.imshow(A[:,order][order,:], cmap=cmap)        
        else:
            plt.imshow(A, cmap=cmap)
        plt.colorbar()
        plt.show()
        
class WDCSBM(WSBM):
    ''' Weighted degree-corected stochastic block model.
    
    Parameters:
    
    theta (torch.Tensor, size: number of classes):
        classes probabilities.
    B (torch.Tensor, size: number of classes * number of classes): 
        edges probabilities between nodes of specific classes.
    z (torch.Tensor, size: number of classes * number of nodes): 
        nodes assignment to a specific class.
    A (torch.Tensor, size: number of nodes * number of nodes):
        weighted adjacency matrix.
    delta_distr (torch.Tensor, size: number of classes * 2):
        [mean, scale] of Normal distribution of the expected degree of nodes
        of each class. Means lay approx. in (-6,6).
    g_mu (torch.Tensor, size: number of classes * number of classes): 
        mean weights of edges between nodes of specific classes.
    g_tau (torch.Tensor, size: number of classes * number of classes): 
        precision weights of edges between nodes of specific classes.
        
    Example:

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
        
    '''
    
    def __init__(self, theta, B, delta_distr, g_mu, g_tau):
        self.theta = theta 
        self.B = B
        self.delta_distr = delta_distr
        self.g_mu = g_mu
        self.g_tau = g_tau
        self.z = None
        self.A = None
        
    def generate(self, n, directed=True, weight_interval=None):
        ''' Generate the adjacence matrix A and nodes assignments z.'''
        z = Multinomial(1, self.theta).sample(sample_shape=[n])
        delta_classes = Normal(self.delta_distr.t()[0], self.delta_distr.t()[1]) \
                        .sample(sample_shape=[n])
        self.delta = (delta_classes*z).sum(dim=1)
        A_prob = torch.mm(torch.mm(z,self.B), z.t()) \
                 * torch.sigmoid(self.delta.unsqueeze(-1) \
                                  + self.delta.unsqueeze(-1).t())
        x = self.sample_A(A_prob, directed)  
        weights_loc = torch.mm(torch.mm(z,self.g_mu), z.t())
        weights_scale = 1/torch.mm(torch.mm(z,self.g_tau), z.t()).pow(0.5)
        w = LogNormal(weights_loc,weights_scale).sample()
        if not weight_interval is None:
            w = torch.where(w < weight_interval[0], 
                            torch.ones(w.size())*weight_interval[0], 
                            w)
            w = torch.where(w > weight_interval[1], 
                            torch.ones(w.size())*weight_interval[1], 
                            w)
        if not directed:
            w = w.triu(diagonal=1) + w.triu(diagonal=1).t()
        A = x*w
        self.z, self.A = z, A        
        return(self.z, self.A)       
        
class WCRG(SBM):
    ''' Weighted complete random graph.
    
    Parameters:
    
    theta (torch.Tensor, size: number of classes):
        classes probabilities.
    w (torch.Tensor, size: number of classes * number of classes): 
        edges probabilities between nodes of specific classes.
    z (torch.Tensor, size: number of classes * number of nodes): 
        nodes assignment to a specific class.
    A (torch.Tensor, size: number of nodes * number of nodes):
        weighted adjacency matrix.
    delta_distr (torch.Tensor, size: number of classes * 2):
        [mean, scale] of Normal distribution of the expected degree of nodes
        of each class. Means lay approx. in (-6,6).
    g_mu (torch.Tensor, size: number of classes * number of classes): 
        mean weights of edges between nodes of specific classes.
    g_tau (torch.Tensor, size: number of classes * number of classes): 
        precision weights of edges between nodes of specific classes.
        
    Example:

    N = 75
    p = torch.tensor([0.2, 0.3, 0.5])
    delta = torch.tensor([[0.,1.], [2.,1.], [-2.,4.]])
    g_mu = torch.tensor([
            [10., 5., 2.],
            [5., 50., 2.],
            [2., 2., 20.]])
    g_tau = torch.ones([3,3])*2
    model = WCRG(p, g_mu.log(), g_tau) 
    z, A = model.generate(N, directed=True)
    model.show(sorted=True)       
        
    '''
    
    def __init__(self, theta, g_mu, g_tau):
        self.theta = theta 
        
        self.g_mu = g_mu
        self.g_tau = g_tau
        self.z = None
        self.A = None
        
    def generate(self, n, directed=True, weight_interval=None):
        ''' Generate the adjacence matrix A and nodes assignments z.'''
        z = Multinomial(1, self.theta).sample(sample_shape=[n])        
        weights_loc = torch.mm(torch.mm(z,self.g_mu), z.t())
        weights_scale = 1/torch.mm(torch.mm(z,self.g_tau), z.t()).pow(0.5)
        w = LogNormal(weights_loc,weights_scale).sample()
        if not weight_interval is None:
            w = torch.where(w < weight_interval[0], 
                            torch.ones(w.size())*weight_interval[0], 
                            w)
            w = torch.where(w > weight_interval[1], 
                            torch.ones(w.size())*weight_interval[1], 
                            w)
        if not directed:
            w = w.triu(diagonal=1) + w.triu(diagonal=1).t()
        self.z, self.A = z, w        
        return(self.z, self.A)         
        
###############################################################################

class HRG():
    ''' Hyperbolic random graph.
    
    Parameters:
        
    R    
    alpha
    T
    A (torch.Tensor, size: number of nodes * number of nodes):
        adjacency matrix.
        
    Example:

    N = 200
    G = HRG(R=6.0,
            alpha=1.5,
            T=0.1)
    r, theta, A = G.generate(N)
    G.show()
    G.plot()
    '''
    
    def __init__(self, R, alpha, T, dtype=torch.double):
        self.dtype = dtype
        self.R = torch.tensor(R).to(self.dtype)
        self.alpha = torch.tensor(alpha).to(self.dtype)
        self.T = torch.tensor(T).to(self.dtype)
        self.A = None
        
    def transform(self, r): 
        return arcosh(1 + ((self.alpha*self.R).cosh()-1)*r)/self.alpha
            
    def generate(self, n, r=None, theta=None, directed=False):
        ''' Generate the adjacency matrix A and the nodes attribute u.'''
        u = Uniform(0.,1.).sample([n]).to(self.dtype)
        if r is None:
            self.r = self.transform(u)
        else:
            self.r = r.to(self.dtype)
        if theta is None:
            self.theta = Uniform(0., 2*np.pi).sample([n]).to(self.dtype)
        else:
            self.theta = theta.to(self.dtype)
        M = torch.ones([n, n]).to(self.dtype)
        Mr = M * self.r
        Mt = M * self.theta
        Md = hyperdist(Mr, Mr.t(), Mt, Mt.t())
        #Md = undirect(Md)
        if self.T==0:
            W = undirect((Md<self.R).float())
            print(W)
        else:
            W = 1/(1+((Md-self.R)/(2*self.T)).exp())
        self.A = self.sample_A(W, directed=False)
        return self.r, self.theta, self.A

    def generate_W(self, n, r=None, theta=None, directed=False):
        ''' '''
        u = Uniform(0.,1.).sample([n]).to(self.dtype)
        if r is None:
            self.r = self.transform(u)
        else:
            self.r = r.to(self.dtype)
        if theta is None:
            self.theta = Uniform(0., 2*np.pi).sample([n]).to(self.dtype)
        else:
            self.theta = theta.to(self.dtype)
        M = torch.ones([n, n]).to(self.dtype)
        Mr = M * self.r
        Mt = M * self.theta
        Md = hyperdist(Mr, Mr.t(), Mt, Mt.t())
        Md = undirect(Md)
        W = (Md-self.R)/(2*self.T)
        
        return W
    
    def sample_A(self, W, directed=False):
        ''' Sample the adjacence matrix A from Bernoulli distribution
        with probabilities W.'''
        A = Bernoulli(W).sample()        
        if not directed:
            A = undirect(A)
        else:
            A = (1. - torch.eye(A.shape[0]))*A        
        return(A)
        
        
    def show_W(self, u=None, size=(6,6), cmap='viridis'):
        ''' Plot the edges probability matrix.'''
        if u is None:
            u = self.u
        w = self.W(u, u)
        plt.figure(figsize=size)
        plt.imshow(w, cmap=cmap)
        plt.colorbar()
        plt.show()
        
    def show(self, A=None, size=(6,6), cmap='viridis'):
        ''' Plot the adjacency matrix; if sorted: accordingly to u. '''
        
        if A is None:
            A = self.A        
        plt.figure(figsize=size)
        plt.imshow(A, cmap=cmap)
        plt.colorbar()
        plt.show()
        
    def plot(self, figsize=(6,6)):
        nodes = torch.stack((self.r,self.theta), dim=1)
        A_ = self.A.triu(diagonal=1)
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111, projection='polar')
        ax.scatter(nodes[:,1].numpy(), nodes[:,0].numpy(), color='black', alpha=0.7, s=6)
        for link in A_.nonzero():
            ax.plot(nodes[link,1].numpy(), nodes[link,0].numpy(), color='gray', alpha=0.2)
        ax.set_rmax(self.R.max()*1.02)
        ax.set_rticks([]) 
        ax.set_axis_off()
        plt.show() 

###############################################################################
        
class EdgesIterableDataset(IterableDataset):
    ''' Dataset class; transformes the given adjacency matrix to 
    a list of edges and allowes to iterate over them.
    
    Parameters:
    
    A (torch.Tensor, size: number of nodes * number of nodes):
        adjacency matrix.
    edges (triple of torch.Tensor):
        list of edges; each edge defined as (start node {torch.int}, 
        finish node {torch.int}, weight {torch.float}).
        
    Example (with torch.utils.data.DataLoader):
    
    >> dataloader = DataLoader(EdgesDataset(A), 
                               batch_size=10, 
                               shuffle=True, 
                               num_workers=0)
    >> dataiter = iter(dataloader)
    >> idx1, idx2, data = dataiter.next()
    '''
    
    def __init__(self, adj_matrix):
        super(EdgesIterableDataset).__init__()
        assert adj_matrix.size()[0]==adj_matrix.size()[1]
        self.A = adj_matrix
        edges = []
        for i in range(self.A.size()[0]):
            for j in range(self.A.size()[1]):
                edges.append((i,j,self.A[i,j]))
        self.edges = edges
        
#    def __len__(self):
#        return len(self.edges)
#    
#    def __getitem__(self, idx):
#        return self.edges[idx]
        
    def __iter__(self):
        return iter(self.edges)
        
    def get_matrix(self):
        return self.A

class EdgesDataset(Dataset):
    ''' Dataset class; transformes the given adjacency matrix to 
    a list of edges and allowes to iterate over them.
    
    Parameters:
    
    A (torch.Tensor, size: number of nodes * number of nodes):
        adjacency matrix.
    edges (triple of torch.Tensor):
        list of edges; each edge defined as (start node {torch.int}, 
        finish node {torch.int}, weight {torch.float}).
        
    Example (with torch.utils.data.DataLoader):
    
    >> dataloader = DataLoader(EdgesDataset(A), 
                               batch_size=10, 
                               shuffle=True, 
                               num_workers=0)
    >> dataiter = iter(dataloader)
    >> idx1, idx2, data = dataiter.next()
    '''
    
    def __init__(self, adj_matrix, directed=True, diagonal=True):
        super(EdgesDataset).__init__()
        assert adj_matrix.size()[0]==adj_matrix.size()[1]
        self.A = adj_matrix
        edges = []
        for i in range(self.A.size()[0]):
            for j in range(self.A.size()[1]):
                if directed:
                    edges.append((i,j,self.A[i,j]))
                else:
                    if i<j or (i==j and diagonal):
                        edges.append((i,j,self.A[i,j]))
        self.edges = edges
        
    def __len__(self):
        return len(self.edges)
    
    def __getitem__(self, idx):
        return self.edges[idx]
        