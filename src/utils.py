# -*- coding: utf-8 -*-
"""
Additional functions
"""
import torch
import warnings
import itertools
import copy

def diriKL(alphas, betas):
    ''' Kullback-Leibler Divergence Between Two Dirichlet Distributions.    
    See http://bariskurt.com/kullback-leibler-divergence-between-two-dirichlet-and-beta-distributions/
    K - number of classes.
    d - number of parameters of the disrtibutions.
    
    Arguments:
    
    alphas (torch.Tensor, size: d or K*K*d): 
        parameters of the first Dirichlet disrtibution 
        or matrix K*K of parameters. 
    betas (torch.Tensor, size: d or K*K*d): 
        parameters of the second Dirichlet disrtibution
        or matrix K*K of parameters.
    '''   
    dim = len(alphas.size())-1
    alpha_0 = alphas.sum(dim=dim)
    beta_0 = betas.sum(dim=dim)
    kl = alpha_0.lgamma() - alphas.lgamma().sum(dim=dim)\
        - beta_0.lgamma() + betas.lgamma().sum(dim=dim) \
         + ((alphas - betas)*(alphas.digamma() \
         - alpha_0.digamma().unsqueeze(dim))).sum(dim=dim)
    return kl
    
def normKL(dist1, dist2):
    if len(dist1.shape)==1:
        mu1, sig1, mu2, sig2 = dist1[0], dist1[1], dist2[0], dist2[1]
    elif len(dist1.shape)==2:
        mu1, sig1, mu2, sig2 = dist1[:,0], dist1[:,1], dist2[:,0], dist2[:,1]
    elif len(dist1.shape)==3:
        mu1, sig1, mu2, sig2 = dist1[:,:,0], dist1[:,:,1], dist2[:,:,0], dist2[:,:,1]
        
    tmp = 0.5*(mu1-mu2).pow(2)/sig2.pow(2)\
        + 0.5*(sig1/sig2).pow(2) + sig2.log() - sig1.log() - 0.5
    if torch.isnan(tmp).sum():
        #print(tmp)
        return torch.zeros(tmp.shape)
    else:
        return tmp
    
def normKLv(dist1, dist2):
    mu1, sig1 = dist1[:,0], dist1[:,1]
    mu2, sig2 = dist2[:,0], dist2[:,1]
    tmp = 0.5*(mu1-mu2).pow(2)/sig2.pow(2)\
        + 0.5*(sig1/sig2).pow(2) + sig2.log() - sig1.log() - 0.5 
    return tmp    
    
def gammaKL(gamma_1, gamma_2):
    ''' Kullback-Leibler Divergence Between Two Gamma Distributions.    
    See https://stats.stackexchange.com/questions/11646/kullback-leibler-divergence-between-two-gamma-distributions
    K - number of classes.
    d - number of parameters of the disrtibutions.
    
    Arguments:
    
    '''
    if len(gamma_1.shape)==1:
        gamma_10, gamma_11, gamma_20, gamma_21 = gamma_1[0], gamma_1[1], gamma_2[0], gamma_2[1]
    elif len(gamma_1.shape)==2:
        gamma_10, gamma_11, gamma_20, gamma_21 = gamma_1[:,0], gamma_1[:,1], gamma_2[:,0], gamma_2[:,1]
    elif len(gamma_1.shape)==3:
        gamma_10, gamma_11, gamma_20, gamma_21 = gamma_1[:,:,0], gamma_1[:,:,1], gamma_2[:,:,0], gamma_2[:,:,1]
        
    def I(a,b,c,d):    
        return - c * d / a - b * a.log() - b.lgamma()\
                + (b-1)*(d.digamma() + c.log())    
    tmp = I(gamma_11,gamma_10,gamma_11,gamma_10) - I(gamma_21,gamma_20,gamma_11,gamma_10)
#    if torch.isinf(tmp).sum():
#        print(tmp)
#        return torch.zeros(tmp.shape)
#    else:
    return tmp
            
def gammaKLalt(gamma_1, gamma_2):
    ''' Kullback-Leibler Divergence Between Two Gamma Distributions.    
    See https://stats.stackexchange.com/questions/11646/kullback-leibler-divergence-between-two-gamma-distributions
    K - number of classes.
    d - number of parameters of the disrtibutions.
    
    Arguments:
    
    '''
    cq = gamma_1[0]
    bq = gamma_1[1]
    cp = gamma_2[0]
    bp = gamma_2[1]
    return (cq - 1)*cq.digamma() - bq.log() - cq - cq.lgamma() \
        + cp.lgamma() + cp*bp.log() - (cp-1)*(cq.digamma() + bq.log()) + bq*cq/bp
            
def gammaKLv(gamma_1, gamma_2):
    ''' Kullback-Leibler Divergence Between Two Gamma Distributions.    
    See https://stats.stackexchange.com/questions/11646/kullback-leibler-divergence-between-two-gamma-distributions
    K - number of classes.
    d - number of parameters of the disrtibutions.
    
    Arguments:
    
    '''
    def I(a,b,c,d):    
        return - c * d / a - b * a.log() - b.lgamma()\
                + (b-1)*(d.digamma() + c.log())
    return I(gamma_1[:,1],gamma_1[:,0],gamma_1[:,1],gamma_1[:,0]) \
            - I(gamma_2[:,1],gamma_2[:,0],gamma_1[:,1],gamma_1[:,0])
            
def warn_tensor(tensor, variable):
    '''Raise a warning if 'tensor' has NaN or inf in it, returns 'tensor'.
    tensor : torch.Tensor
    variable (str): name of the tensor. 
    '''
    if torch.isnan(tensor).sum()>0:
        warnings.warn(str('%s has NaN in it!' % variable))
    if torch.isinf(tensor).sum()>0:
        warnings.warn(str('%s has inf in it!' % variable))
    if (tensor==0).sum():
        warnings.warn(str('%s has 0 in it!' % variable))
    if (tensor==1).sum():
        warnings.warn(str('%s has 1 in it!' % variable))
    return tensor
    
def bad_tensor(tensor):
    '''Raise a warning if 'tensor' has NaN or inf in it, returns 'tensor'.
    tensor : torch.Tensor
    variable (str): name of the tensor. 
    '''
    return bool(torch.isnan(tensor).sum() + torch.isinf(tensor).sum())
    
def class_compare(eta1, eta2):
    z2 = eta2.argmax(dim=0).float()
    z1 = eta1.argmax(dim=0).float()
    z1_unique = z1.unique()
    z2_unique = z2.unique()
    max_class = max(len(z1_unique), len(z2_unique))
    for i in range(len(z1_unique)):
        z1 = torch.where(z1==z1_unique[i], 
                         torch.ones(z1.size())*i,
                         z1)
    for i in range(len(z2_unique)):
        z2 = torch.where(z2==z2_unique[i], 
                         torch.ones(z2.size())*i,
                         z2)
    permutations = itertools.permutations(range(max_class))
    z2_modify = z2 + max_class
    best_score, best_z2, best_comp, best_perm = 0, 0, 0, 0
    for p in permutations:
        tmp = z2_modify.clone()
        for i in range(max_class):
            tmp = torch.where(tmp==(max_class+i), 
                              torch.ones(tmp.size())*p[i], 
                              tmp)
        comp = tmp==z1
        if best_score < comp.sum():
            best_score=comp.sum()
            best_z2 = tmp.clone()
            best_comp = comp.clone()
            best_perm = copy.deepcopy(p)
    same_num = best_comp.sum()
    print(same_num)
    diff_id = best_comp.argsort()[:-same_num]
    
    return best_score, z1, best_z2, diff_id.sort().values, best_perm
    
arcosh = lambda x: (x + (x**2 - 1).sqrt()).log()

hyperdist = lambda rx,ry,fx,fy: arcosh(rx.cosh()*ry.cosh() - rx.sinh()*ry.sinh()*(fx-fy).cos())

p_hd = lambda rx,ry,fx,fy,R,T: 1/(1+((hyperdist(rx,ry,fx,fy)-R)/(2*T)).exp())

undirect = lambda A: A.triu(diagonal=1) + A.triu(diagonal=1).t()

# https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
log1mexp = lambda a: torch.where(a>torch.tensor(2.).to(a.dtype).log(),
                                torch.log1p(-torch.exp(-a)),
                                torch.log(-torch.expm1(-a)))

def cart2polar(x, y):
    """
    Transform Cartesian coordinates to polar.
    Parameters
    ----------
    x, y : floats or arrays
        Cartesian coordinates
    Returns
    -------
    r, theta : floats or arrays
        Polar coordinates
    """
    r = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y,x)  # θ referenced to vertical
    return r, theta
    
def c2d(x):
    """
    Transform Cartesian coordinates to polar degree (r=1).
    Parameters
    ----------
    x, y : floats or arrays
        Cartesian coordinates
    Returns
    -------
    theta : floats or arrays
        Polar degree
    """
    x0, x1 = x.select(-1,0), x.select(-1,1)
    return torch.atan2(x1,x0)  # θ referenced to vertical

def polar2cart(r, theta):
    """
    Transform polar coordinates to Cartesian.
    Parameters
    ----------
    r, theta : floats or arrays
        Polar coordinates
    Returns
    -------
    [x, y] : floats or arrays
        Cartesian coordinates    
    """
    return torch.stack((r * theta.cos(), r * theta.sin()), dim=-1).squeeze()
    