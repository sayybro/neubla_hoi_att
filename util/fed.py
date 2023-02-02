# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import json
import numpy as np
from torch.nn import functional as F

def load_class_freq(
    path='data/vaw/annotations/vaw_train_cat_info.json', freq_weight=1.0):
    cat_info = json.load(open(path, 'r'))
    cat_info = torch.tensor(
        [c['image_count'] for c in sorted(cat_info, key=lambda x: x['id'])])
    freq_weight = cat_info.float() ** freq_weight
    return freq_weight


def get_fed_loss_inds(gt_classes, num_sample_cats, C, weight=None):
    #gt_classes : torch.cat([pos_gt_classes,neg_gt_classes])
    #num_sample_cats : 50
    #weight : self.fed_loss_weight,  #torch.Size([620]) 
    #C : 620
    appeared = torch.unique(gt_classes) #torch.Size([29])
    prob = appeared.new_ones(C + 1).float() #torch.Size([621]),Returns a Tensor of size size filled with 1
    prob[-1] = 0
    if len(appeared) < num_sample_cats: #pos,neg label 수가 num_sample_cats보다 작으면 
        if weight is not None:
            prob[:C] = weight.float().clone() 
        prob[appeared] = 0 #appeared에 있는 category는 prob 0으로 
        more_appeared = torch.multinomial( # prob: the input tensor containing probabilities
            prob, num_sample_cats - len(appeared), #50-29=21개만큼 multinomial에서 sampling
            replacement=False) 
        #Returns a tensor where each row contains num_samples indices sampled 
        #from the multinomial probability distribution located in the corresponding row of tensor input
        appeared = torch.cat([appeared, more_appeared])
    return appeared
