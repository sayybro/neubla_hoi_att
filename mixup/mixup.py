import torch
import numpy as np
import random

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''

    if alpha > 0:
        lam = np.random.beta(alpha, alpha) #Draw samples from a Beta distribution
    else:
        lam = 1

    count_y = [len(label)for label in y]
    max_y = max(count_y) 

    duplicate_x = [x_sample.expand(max_y,x_sample.shape[0],x_sample.shape[1],x_sample.shape[2]).cpu().detach() for x_sample, count in zip(x,count_y)]
    batch_size = x.size()[0]    
    indices = torch.randperm(batch_size)
    x_b = [duplicate_x[index] for index in indices] 
    mixed_x = [lam * x_dup_sample + (1 - lam) * x_b_sample for x_dup_sample, x_b_sample in zip(duplicate_x,x_b)]
    
    #y_a: original
    #y_b: permuted
    y_a, y_b = y, [y[index] for index in indices]

    mixed_x_a = [random.sample(list(sample),len(n_y_a)) for sample, n_y_a in zip(mixed_x,y_a)] 
    mixed_x_b = [random.sample(list(sample),len(n_y_b)) for sample, n_y_b in zip(mixed_x,y_b)]     
    return mixed_x_a, mixed_x_b, y_a, y_b, lam

def mixup_criterion(criterion, pred_a, pred_b ,y_a, y_b, lam):
    return lam * criterion(pred_a, y_a) + (1 - lam) * criterion(pred_b, y_b)