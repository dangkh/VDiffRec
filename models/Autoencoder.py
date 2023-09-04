import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math
from torch.nn.init import xavier_normal_, constant_, xavier_uniform_
from kmeans_pytorch import kmeans

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AutoEncoder(nn.Module):
    """
    Guassian Diffusion for large-scale recommendation.
    """
    def __init__(self, item_emb, in_dims, device, dropout=0.1):
        super(AutoEncoder, self).__init__()

        self.item_emb = item_emb
        self.maxItem = 1000
        self.in_dims = in_dims
        self.n_item = len(item_emb)
        self.dropout = nn.Dropout(dropout)

        self.reduceDim = nn.Linear(self.maxItem * 64, self.in_dims)
        # self.decodeDim = nn.Linear(self.in_dims[0], self.maxItem * 64)
        self.predictItem = nn.Linear(self.in_dims, self.n_item)
        self.activateF = nn.Sigmoid()
        self.loss = torch.nn.MSELoss()
        # self.apply(xavier_normal_initialization)
    

    def Encode(self, batch):
        batch = self.dropout(batch)
        batch = self.reduceDim(batch)

        return '', batch, ''

        
    def Decode(self, batch):
        return self.activateF(self.predictItem(batch))

def compute_loss(recon_x, x):
    return torch.nn.MSELoss()(recon_x, x)
    # return -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))  # multinomial log likelihood in MultVAE


# def xavier_normal_initialization(module):
                