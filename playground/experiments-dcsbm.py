# -*- coding: utf-8 -*-
"""

@author: mozzhorin

python3 experiments-dcsbm.py --lr=[0.1,0.05,0.01] --epochs=5 --trials=2 --gpu=0 --threads=4 
"""

#import pandas as pd
import numpy as np
import pandas as pd
import argparse
import time
import sys
sys.path.append('../src/')
from vi_sbm import * 
import pickle
from torch import autograd
from graph_models import SBM, DCSBM, WDCSBM, EdgesDataset
torch.manual_seed(55)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit SBM to EBA data.')

#    parser.add_argument('--data', type=str, default='../data/EBA_Pi.csv', 
#                        help='Adjacency matrix') 
#    parser.add_argument('--name', type=str, default='eba_pi', 
#                        help='Name') 
    parser.add_argument('--N', type=int, default=200, 
                        help='Number of vertices')
#    parser.add_argument('--classes', type=int, default=3, 
#                        help='Number of classes')
    parser.add_argument('--epochs', type=str, default='5', 
                        help='Number of epochs')
    parser.add_argument('--lr', type=str, default='0.1', 
                        help='Learning rate')
    parser.add_argument('--trials', type=int, default=1, 
                        help='Number of trials')
    parser.add_argument('--batch', type=int, default=256, 
                        help='Batch size')
    parser.add_argument('--workers', type=int, default=0, 
                        help='Number of workers for the dataloader')
    parser.add_argument('--gpu', type=int, default=0, 
                        help='GPU usage')
    parser.add_argument('--threads', type=int, default=-1, 
                        help='Number of threads')
    args = parser.parse_args()
                        
    if args.threads > 0:                         
        torch.set_num_threads(args.threads)       
    
    print('>>> Generate the data...')
    num_classes = 3                      # Number of classes
    p = torch.tensor([0.2, 0.3, 0.5])    # Probability of each class
    B = torch.tensor([                   # Connection probability between classes
        [0.8, 0.1, 0.3],
        [0.1, 0.9, 0.1],
        [0.5, 0.1, 0.8]])
    delta = torch.tensor(
            [[0.,1.], [0.,1.], [0.,1.]])
    dcsbm = DCSBM(p, B, delta)
    z, A = dcsbm.generate(args.N)
    
    delta_init = torch.ones([args.N,2])
    delta_mu = A.sum(dim=1)/A.sum(dim=1).mean()
    delta_init[:,0]=delta_mu.log().clone()
     
    dataloader = DataLoader(EdgesDataset(A), 
                            batch_size=args.batch, shuffle=True, 
                            num_workers=args.workers)

    if args.gpu:
        if not torch.cuda.is_available():
            print(">>> GPU is not available. Using CPU.")
            device = torch.device("cpu")
        else:
            print(">>> Using GPU.")
            device = torch.device("cuda")
    else:
        print(">>> Using CPU.")
        device = torch.device("cpu")
                            
    t1 = time.time()
    epochs = eval(args.epochs)
    lrs = eval(args.lr)    
        
    vi = VI_DCSBM(num_nodes=args.N, num_classes=num_classes, 
                       init_values={'etas':None, 
                              'thetas':None, 
                              'Bs':None,
                              'deltas':delta_init},
                       device=device).to(device)
    
    print('>>> Fitting DCSBM...')
    vi.multi_train(dataloader, trials=args.trials, epochs=epochs, lrs=lrs)
    t2 = time.time()
    success = 0
    for i in range(len(vi.multi_results[0])):
        if vi.class_accuracy(z,vi.multi_results[0][i]).item()==1:
            success+=1
    total_time = t2-t1
    print('>>> Total time: %.2f, average time: %.2f' \
          % (total_time, total_time/args.trials))
    print('>>> Success rate: %.2f' \
          % (success/args.trials))
