#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:10:34 2020

@author: mo
"""

import sys
sys.path.append('../src/')
from vi_hrg import *
from utils import *
from torch import autograd
torch.manual_seed(52)

def noise_r(x, R, rel_var=0.1, epsilon=1e-4):
    rs = torch.distributions.normal.Normal(x, R*rel_var).sample() 
    return torch.clamp(rs, min=0+epsilon, max=R-epsilon)

def noise_phi(x, rel_var=0.1):
    phis = torch.distributions.normal.Normal(x, 2*np.pi*rel_var).sample()
    return phis % (2*np.pi)

N = 300
R = 5.0
alpha = .55
T = 0.2

G = HRG(R=R, alpha=alpha, T=T)
r, theta, A = G.generate(N)
G.show()
#G.plot()

r_init = noise_r(r, R, rel_var=0.1)
phi_init = noise_phi(theta, 0.1)

r_loc_init = logit(r_init/R)
r_scale_init = (torch.ones([N])/10).log()
phi_loc_init = polar2cart(1, phi_init)
phi_scale_init = (torch.ones([N])*30).log()
R_conc_init = torch.tensor(10.).log()
R_scale_init = torch.tensor(1.).log()
alpha_conc_init = torch.tensor(.5).log()
alpha_scale_init = torch.tensor(.5).log()
T_init = torch.tensor([3.,10.]).log()
dataloader = DataLoader(EdgesDataset(A), batch_size=int(N**2/64),  num_workers=0, shuffle=True)
vi = VI_HRG(N,20, init_values={'rs_loc':r_loc_init,
                                'rs_scale':r_scale_init,
                              'phis_loc':phi_loc_init,
                              'phis_scale':phi_scale_init, 
                              'R_conc':R_conc_init, 
                              'R_scale':R_scale_init,
                              'alpha_conc':alpha_conc_init,
                              'alpha_scale':alpha_scale_init,
                              'T':T_init},
           fixed={'R':None, 
                  'T':None,
                  'alpha':None},
           priors={'R_p':torch.tensor([20., 0.4]), 
                    'T_p':torch.tensor([1., 15.]),
                    'alpha_p':torch.tensor([27., 0.03])},
                   dtype=torch.double)
#print('Train!!!')
#with autograd.detect_anomaly():
#vi.train(dataloader, lrs=[0.1], debug=False, epochs=10)
vi.multi_train(dataloader, lrs=0.1, epochs=5, trials=2)
r_x_loc, r_x_scale, phi_x_loc, phi_x_scale, R_x_conc, R_x_scale, T_x, \
    alpha_x_conc, alpha_x_scale = vi.constrained_params()
hist_samples = [1000]
bins = 20
R_samples = Gamma(R_x_conc, R_x_scale.reciprocal()).sample(hist_samples)
alpha_samples = Gamma(alpha_x_conc, alpha_x_scale.reciprocal()).sample(hist_samples)
T_samples = Beta(T_x[0], T_x[1]).sample(hist_samples)
plt.figure(figsize=(8,12))

plt.subplot(3, 1, 1)
plt.hist(R_samples.numpy(), bins=bins)
plt.axvline(R, color='r')
plt.title('Estimated R')

plt.subplot(3, 1, 2)
plt.hist(alpha_samples.numpy(), bins=bins)
plt.axvline(alpha, color='r')
plt.title('Estimated alpha')

plt.subplot(3, 1, 3)
plt.hist(T_samples.numpy(), bins=bins)
plt.axvline(T, color='r')
plt.title('Estimated T')
plt.show()
print('Likelihood:', vi.likelihood())