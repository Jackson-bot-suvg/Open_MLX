from torch.utils.data import Dataset
import numpy as np
import torch
from torch import nn
import math
from model_utils import * 

import faulthandler
faulthandler.enable()


N_TIME_STEP = 5
NUM_FEAT_SIZE = 9
N_EMBD = 32
N_LAYER = 2
Y_HAT_SIZE = 3
N_ATTRIBUTES = [4, 1]
N_ATTRIBUTES = [3, 1]
N_ATTRIBUTES = [6, 5]

rtm3_lobs = [(0, 0),
             (0, 1),
             (0, 2),
             (0, 3),
             (0, 4),
             (1, 0),
             (1, 1),
             (1, 2),
             (1, 3),
             (1, 4),
             (2, 4),
             (3, 2),
             (3, 3),
             (3, 4),
             (4, 2),
             (4, 3),
             (4, 4),
             (5, 2),
             (5, 3),
             (5, 4)
            ]                               

class Chan(nn.Module):
    def __init__(self, time_step, n_channel, n_embd, n_layer, y_hat_size):
        super().__init__()
        self.blocks = nn.Sequential(
            *[SharedSpecBlock(
                time_step=time_step, 
                n_channel=n_embd, 
                n_embd = n_embd,
                n_shared=8, 
                # n_spec=8, 
                n_spec=16, 
                n_attrs=N_ATTRIBUTES
            ) 
            for i in range(n_layer)]
        )

        self.ln_f = nn.LayerNorm(n_embd * time_step) # final layer norm
        self.fcst_head = nn.Linear(n_embd * time_step, y_hat_size)

    def forward(self, x, x_idx, emb = None):
        B, time_step, n_embd = x.shape     
        tuple_x, _ = self.blocks( ( (x, x_idx), emb ) ) # (B, T, C)
        x, _ = tuple_x
        x = x.view(B, n_embd * time_step)
        x = self.ln_f(x)   # (B, T, C)
        # x = x[:, -Y_HAT_SIZE:, :] + x[:, -Y_HAT_SIZE-1:-Y_HAT_SIZE, :]
        x = self.fcst_head(x)
        return x # (B, H ,y_hat_size)



class SODataset(Dataset):
    def __init__(self, X):    
        self.in_mem_X = X
        self.size = len(X)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x_ctx = torch.tensor(self.in_mem_X.iloc[idx].iloc[:10].values, dtype=torch.int32)
        x = torch.tensor(self.in_mem_X.iloc[idx].iloc[10:-2].values, dtype=torch.float32)# 48 dimension
        y = torch.tensor(self.in_mem_X.iloc[idx].iloc[-1], dtype=torch.float32)
        return (
            x,
            x_ctx,
            y,
        )




class SalesRadar(nn.Module):
    def __init__(
        self,
        rtm_num,
        sub_rtm_num,
        new_pos_type_num,
        state_num,
        city_group_num,
        district_group_num,
        lob_num,
        event_tier_num,
        event_cate_num,
        quantiles,
    ):
        super().__init__()
        self.quantiles = quantiles


        self.rtm_embed = nn.Embedding(rtm_num, N_EMBD)
        self.sub_rtm_embed = nn.Embedding(sub_rtm_num, N_EMBD)  
        self.state_embed = nn.Embedding(state_num, N_EMBD)
        self.city_group_embed = nn.Embedding(city_group_num, N_EMBD)
        self.district_group_embed = nn.Embedding(district_group_num, N_EMBD)
        self.lob_embed = nn.Embedding(lob_num, N_EMBD)
        # self.model_embed = nn.Embedding(model_num, N_EMBD)
        self.event_tier_embed = nn.Embedding(event_tier_num, N_EMBD)
        self.event_cate_embed = nn.Embedding(event_cate_num, N_EMBD)  
        # self.model_more_embed = nn.Linear(3, N_EMBD)
        self.npi_embed = nn.Linear(1, N_EMBD)
        self.new_pos_type_embed = nn.Embedding(new_pos_type_num, N_EMBD)  

        self.log1p_exp = Log1p_Exp()


        self.norm_proj1 = RevIN(NUM_FEAT_SIZE)
        self.norm_proj2 = RevIN(NUM_FEAT_SIZE)
        # self.norm_proj3 = RevIN(NUM_FEAT_SIZE)
        self.conv_proj1 = nn.Conv1d(NUM_FEAT_SIZE, N_EMBD, kernel_size=3, dilation=1)
        self.conv_proj2 = nn.Conv1d(NUM_FEAT_SIZE, N_EMBD, kernel_size=2, dilation=1)


        self.ll_proj = nn.Sequential(
            *[
                nn.LayerNorm(NUM_FEAT_SIZE),
                nn.Linear(NUM_FEAT_SIZE, N_EMBD),
                nn.ReLU(),
                nn.LayerNorm(N_EMBD),
                nn.Linear(N_EMBD, N_EMBD),
            ]
        )

        self.emb_proj = nn.Sequential(
            *[
                nn.Linear(1, N_TIME_STEP),
                nn.ReLU(),
                nn.LayerNorm(N_TIME_STEP),
                nn.Linear(N_TIME_STEP, N_TIME_STEP),
            ]
        )        

        self.chan1 = Chan(N_TIME_STEP, NUM_FEAT_SIZE, N_EMBD, N_LAYER, Y_HAT_SIZE)          
        self.chan2 = Chan(N_TIME_STEP, NUM_FEAT_SIZE, N_EMBD, N_LAYER, Y_HAT_SIZE) 
        self.quantiles = quantiles
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x_idx,        
        x,
        x_ctx,
        y,
    ):

        # --- context feats --------------------------------------
        B, _ = x_ctx.shape
        x_ctx = x_ctx.to(DEVICE)
        y = y.to(DEVICE)
        rtm_emb = self.rtm_embed(x_ctx[:, 0])
        sub_rtm_emb = self.sub_rtm_embed(x_ctx[:, 1])
        state_emb = self.state_embed(x_ctx[:, 2])
        city_group_emb = self.city_group_embed(x_ctx[:, 3])
        district_group_emb = self.district_group_embed(x_ctx[:, 4])
        lob_emb = self.lob_embed(x_ctx[:, 5])
        event_tier_emb = self.event_tier_embed(x_ctx[:, 6])
        event_cate_emb = self.event_cate_embed(x_ctx[:, 7])
        npi_emb = self.npi_embed(x_ctx[:, 8].view(B, 1).type(torch.float32))
        new_pos_type_emb = self.new_pos_type_embed(x_ctx[:, 9])
        ctx_emb = state_emb + city_group_emb + district_group_emb + lob_emb + event_tier_emb + event_cate_emb + npi_emb

        ctx_emb = self.emb_proj(ctx_emb.view(B, N_EMBD, 1)).transpose(-2, -1)
        x = x.to(DEVICE).view(B, 5, 9)
        dn_idx = torch.tensor([0,]*Y_HAT_SIZE, device=DEVICE)        

        x1 = x
        x1 = self.norm_proj1(x1, "norm")
        x1 = self.ll_proj(x1)

        y_hat1 = self.chan1(x1, x_idx, ctx_emb).view(B, Y_HAT_SIZE)        
        y_hat1 = self.log1p_exp(y_hat1)
        y_hat1 = self.norm_proj1(y_hat1, "denorm", dn_idx)

        y_hat = y_hat1.view(B, Y_HAT_SIZE)
        rtm3_lobs.index(tuple(x_idx))
        loss = self.loss(y_hat, y, self.quantiles[rtm3_lobs.index(tuple(x_idx))])
        return y_hat, loss



    def loss(self, preds, target, quantiles):
        assert preds.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(quantiles):
            errors = target - preds[:, i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss


if __name__ == "__main__":

    radar = SalesRadar(        
        rtm_num = 3,
        sub_rtm_num = 9,
        state_num = 31,
        city_group_num = 5,
        district_group_num = 13,
        lob_num = 5,
        model_num = 17,
        event_tier_num = 7,
        event_cate_num = 5,
        quantiles = [0.9, 0.95, 0.99],
    )
    # radar = nn.DataParallel(nphone, device_ids=[0]).to(DEVICE)
    print(f'Number of parameters: {sum(p.numel() for p in radar.parameters())}')
    print(f'Device: {DEVICE}')

    x = torch.randn([10, 6, 8], dtype=torch.float64)
    x_ctx = torch.zeros([10, 12], dtype=torch.int32)
    y = torch.randn([10, 1], dtype=torch.float64)
    y_hat, loss = radar([2, 1], x, x_ctx, y)
    print(f"{loss=}")


