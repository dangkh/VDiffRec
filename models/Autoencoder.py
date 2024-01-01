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
    def __init__(self, item_emb, n_cate, in_dims, out_dims, device, act_func, maxItem, reparam=True, dropout=0.1):
        super(AutoEncoder, self).__init__()

        self.item_emb = item_emb.to(device)
        self.maxItem = maxItem
        self.n_cate = n_cate
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.act_func = act_func
        self.n_item = len(item_emb)
        self.reparam = reparam
        self.dropout = nn.Dropout(dropout)
        # self.U = nn.Embedding(self.maxItem * 64, 64, max_norm=True)
        self.reduceDim = nn.Linear( 64, self.in_dims[0])
        self.f1 = nn.Linear( self.in_dims[0], self.in_dims[0])
        # self.decodeDim = nn.Linear(self.in_dims[0], self.maxItem * 64)
        self.predictItem = nn.Linear(self.in_dims[0], self.n_item)
        self.activateF = nn.Sigmoid()
        self.loss = torch.nn.MSELoss()
        self.Tanh = nn.ReLU()
        self.multihead_attn = nn.MultiheadAttention(64, 1)
        self.apply(xavier_normal_initialization)
    

    def Encode(self, batch, label):
        attn_output, _ = self.multihead_attn(batch, batch, batch)
        eLabel = label.unsqueeze(-1)
        attn_output = attn_output * eLabel
        sumAtt = attn_output.sum(1) 
        numI = label.sum(1).reshape(-1, 1)
        batch = sumAtt / numI
        batch = self.dropout(batch)
        batch = self.reduceDim(batch)
        batch = self.Tanh(batch)
        batch = self.dropout(batch)
        batch = self.f1(batch)
        # batch = self.Tanh(batch)
        # batch = self.dropout(batch)
        # batch = self.U(batch)

        return '', batch, ''
    
    def reparamterization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def Decode(self, batch):
        # return self.activateF(torch.matmul(batch, self.item_emb.T))
        return self.activateF(self.predictItem(batch))
    
def compute_loss(recon_x, x):
    mask = torch.where(x!= 0)
    return torch.nn.MSELoss()(recon_x[mask], x[mask])
    # return -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))  # multinomial log likelihood in MultVAE


def xavier_normal_initialization(module):
    r""" using `xavier_normal_`_ in PyTorch to initialize the parameters in
    nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.
    .. _`xavier_normal_`:
        https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_normal_#torch.nn.init.xavier_normal_
    Examples:
        >>> self.apply(xavier_normal_initialization)
    """
    if isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)            
                