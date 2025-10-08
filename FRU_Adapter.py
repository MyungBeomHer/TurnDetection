from functools import partial

import torch
import torch.nn as nn
from torch.autograd import Function

# import timm.models.vision_transformer
from einops import rearrange
import math

class FRU_Adapter(nn.Module):
    def __init__(self,
                 channel = 197,
                 embded_dim = 384,
                 Frame = 400,
                 hidden_dim = 128):
        super().__init__()

        self.Frame = Frame

        self.linear1 = nn.Linear(embded_dim ,hidden_dim)
        self.linear2 = nn.Linear(hidden_dim,embded_dim)

        self.T_linear1 = nn.Linear(Frame, Frame)
        self.softmax = nn.Softmax(dim=1)
        self.ln = nn.LayerNorm(hidden_dim)
        
        self.TFormer = TemporalTransformer(frame=Frame,emb_dim=hidden_dim)

    #Frame recalibration unit
    def FRU(self, x):
        x1 = x.mean(-1).flatten(1) # bn t 
        x1 = self.T_linear1(x1) # bn t
        x1 = self.softmax(x1).unsqueeze(-1) #bn t 1

        x = x * x1 #bn t d
        return x 
    
    def forward(self, x):

        squeezed = False
        if x.dim() == 2:    # T D 
            x = x.unsqueeze(0)
            squeezed = True

        x = self.linear1(x) # bn t d
        x = self.ln(x) 

        # _, _,down = x.shape

        # x = rearrange(x, '(b n) t d-> b t (n d)', t = self.Frame, n = n, d = down)
        # x1 = x.mean(-1).flatten(1) # bn t 
        # x1 = self.T_linear1(x1) # bn t
        # x1 = self.softmax(x1).unsqueeze(-1) #bn t 1
        # x = x * x1 #bn t d
        x = self.FRU(x)
        # x = rearrange(x, 'b t (n d)-> (b n) t d', t = self.Frame, n = n, d = down)

        x = self.TFormer(x)
        x = self.linear2(x) # bn t d
        if squeezed == True:
         x = x.squeeze(0)
        #bt n d
        # x = rearrange(x, '(b n) t d-> (b t) n d', t = self.Frame, n = n, d = d)
        return x

class TemporalTransformer(nn.Module):
    def __init__(self, 
                 frame = 16,
                 #channel = 8,
                 emb_dim = 128,
                 ):
        super().__init__()
        
        self.proj_Q = nn.Linear(emb_dim,emb_dim)
        self.proj_K = nn.Linear(emb_dim,emb_dim)
        self.proj_V = nn.Linear(emb_dim,emb_dim)
        self.proj_output = nn.Linear(emb_dim,emb_dim)
        
        self.norm = nn.LayerNorm(emb_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        #B C T H W 
        _,_,E = x.shape
            
        x1 = self.norm(x) 

        q = self.proj_Q(x1)
        k = self.proj_K(x1)
        v = self.proj_V(x1)

        q_scaled = q * math.sqrt(1.0 / float(E))

        attn_output_weights = q_scaled @ k.transpose(-2, -1)
        attn_output_weights = self.softmax(attn_output_weights)
        attn_output = attn_output_weights @ v 
        attn_output = self.proj_output(attn_output) #B T E  where E = C * H * W
        attn_output = attn_output + x 

        return attn_output 

