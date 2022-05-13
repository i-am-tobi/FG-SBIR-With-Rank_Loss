#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 15:18:02 2022

@author: shreee
"""

import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")
#import numpy


def compute_aff(x,y):
    return torch.mm(x,y.t())

def sigmoid(tensor, temp=1.0):
    exponent = - tensor / temp
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1.0 / (1.0+torch.exp(exponent))
    
    return y

class Rank_loss(nn.Module):
    """
    Defining A Loss Function Based On Ranking Similarities
    
    """
    
    def __init__(self,anneal):
        
        super(Rank_loss, self).__init__()
        
        self.anneal = anneal
        
    def forward(self, query_feature, preds):
        """
        
        preds --> batch_size X feat_dims
        
        """
        #bs = self.batch_size
        sim_all = []
        top_k = []
        for m in query_feature.shape[0]:
            for i in preds.shape[0]:
            
                sim_all[i] = compute_aff(query_feature[m,:],preds[i,:])
        
                sim_sg = sigmoid(sim_all,temp=self.anneal)  ##Ranking in differentible manner
                sim_all_rk = torch.sum(sim_sg, dim=-1) + 1
        
                top_k[m] = sim_all_rk.le(preds.shape[0]).sum().numpy() / sim_all_rk.shape[0]  ##topk accuracy for the first query feature
        
        top_k_avg = top_k.sum().numpy() / top_k.shape[0]
        
        return 1-top_k_avg  ##Returning average top accuracy

        
        
        
        
