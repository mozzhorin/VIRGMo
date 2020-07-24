#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python interface to the Hyperbolic Embedder[1](HE). 

First you have to compile HE from [2]. In my experience, in addition to the
described installation steps, I also have to make several changes:
    - if the error appears:
        main.cpp:69:23: error: ‘time’ is not a member of ‘std’,
      you can add this line to the #include section: 
          #include <ctime>
    - I needed to add the flag -lpthread to the LDFLAGS in makefile
    - HE could not find some libraries although they were installed in 
    /usr/local/lib as required. I had to run:
        sudo ln -s /usr/local/lib/xxx /usr/lib/xxx
    where xxx is the lost library.

[1] T. Bläsius, T. Friedrich, A. Krohmer and S. Laue, 
"Efficient Embedding of Scale-Free Graphs in the Hyperbolic Plane," 
in IEEE/ACM Transactions on Networking, vol. 26, no. 2, pp. 920-933, 
April 2018, doi: 10.1109/TNET.2018.2810186.

[2] https://bitbucket.org/HaiZhung/hyperbolic-embedder/src/master/
"""

import torch
import numpy as np
import pandas as pd
import os

def A2edgelist(A):
    ''' Transforms the unweighted adjacency matrix A to the list of edges.
    
    ARGUMENTS:
        A (torch.tensor, size: N*N): adjacency matrix
        
    RETURNS:
        (list) List of tuples (start_node, finish_node)
    '''
    edges = []
    for i in range(A.size()[0]):
        for j in range(A.size()[1]):
            if A[i,j]==1:
                edges.append((i,j))
    return edges

grad_transform = lambda grad: 2*np.pi*grad/360

def hyperbolic_embedder(A, name='virgmo/hrg_el', seed=32472351):
    
    hypemb_dir = '../../hyperbolic-embedder/'
    python_dir = os.getcwd()
    
    hypemb_cmd = './embedder --logtostderr --input="%s.txt" --seed=%d --embed=%s' % \
            (name, seed, name)
    
    pd.DataFrame(np.array(A2edgelist(A))).to_csv(hypemb_dir+name+'.txt', sep=' ', index=False)
    os.chdir(hypemb_dir)
    os.system(hypemb_cmd)   # Embed
    os.chdir(python_dir)
    
    coord_file = name+'-coordinates.txt'
    df_coord = pd.read_csv(hypemb_dir+coord_file, sep='\t', header=None)
    R_est = float(df_coord[1][1])
    alpha_est = float(df_coord[2][1])
    T_est = float(df_coord[3][1])
    df_coord.drop(3, axis=1, inplace=True)
    df_coord.drop([0,1], axis=0, inplace=True)
    col_coord = ['id', 'r', 'phi']
    df_coord.columns=col_coord
    df_coord['id'] = df_coord['id'].astype('int')
    df_coord.sort_values(by='id', inplace=True)
    
    ids = np.array(df_coord['id'].astype('int'))    
    r_init = torch.tensor(np.array(df_coord['r'].astype('float')))    
    phi_init = grad_transform(torch.tensor(np.array(df_coord['phi'].astype('float'))))
    
    return R_est, alpha_est, T_est, r_init, phi_init, ids