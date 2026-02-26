import torch
import random
import json,os,sys
import numpy as np
import pandas as pd
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, cur_dir)
sys.path.insert(1, cur_dir+'/..')
torch.set_default_dtype(torch.float32)
from model_lob import *

### training_loss / eval_loss / metric

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

traces = [([],[], []) for _ in range(len(rtm3_lobs))]
W_STEP = 2000
F_STEP = 96000
BATCH_SIZE = 128
LEARNING_RATE = 6e-4

index_column = 'idx1'

def print_trace(traces, t_idx):
    back_size = -10
    if t_idx<=0: back_size = -1000
    
    ret_list = [f"{np.mean(traces[i][t_idx][back_size:]):.2f}" for i in range(len(rtm3_lobs))]
    ret = ','.join(ret_list)    
    return ret

def calc_lr(it):
    import math

    if it < W_STEP:
        return LEARNING_RATE * it / W_STEP
    if it > F_STEP:
        return LEARNING_RATE * 0.1

    decay_ratio = (it - W_STEP) / (F_STEP - W_STEP)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return LEARNING_RATE*0.1 + coeff*(LEARNING_RATE - LEARNING_RATE*0.1)

def load_ckp(ckp_path):
    model = SalesRadar()
    checkpoint = torch.load(ckp_path)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix): state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    if torch.cuda.is_available(): model = nn.DataParallel(model, device_ids=[0]).to(DEVICE)
    model = torch.compile(model)

    optim = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, fused=True if torch.cuda.is_available() else False)
    optim.load_state_dict(checkpoint['optimizer'])
    it = checkpoint['iter_num']
    print(f'load model done, model is {ckp_path}', flush=True)

    model.eval()
    print(f'number of parameter is {sum(p.nelement() for p in model.parameters())}', flush=True)
    return model, optim, it


@torch.no_grad()
def fetch_data(data_loader):
    while True:
        for x, x_ctx, y in data_loader:
            yield (x, x_ctx, y)


def quantile_value(y_hat, y, quantile):
    return float(
                np.mean(y.numpy() < y_hat[:, 0].cpu().numpy())
            )

def quantile_loss(preds, target, quantiles):
    assert preds.size(0) == target.size(0)
    losses = []
    preds = preds.to(DEVICE)
    for i, q in enumerate(quantiles):
        target = target.to(DEVICE)
        errors = target - preds[:, i]
        losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
    loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
    return loss


@torch.no_grad()
def estimate_loss(es_model, e_s, rtm3_idx):
    es_model.eval()
    x, x_ctx, y = e_s
    y_hat, _ = es_model(rtm3_idx, x, x_ctx, y)
    loss = quantile_loss(y_hat, y, [0.9, 0.95, 0.99])
    q90 = quantile_value(y_hat, y, 0.9)
    es_model.train()
    return loss.cpu().numpy(), q90

def train(train_df, eval_df, ckp_path=None):
    quantiles = [
        [0.9, 0.95, 0.99], 
        [0.9, 0.95, 0.99], 
        [0.9, 0.95, 0.99], 
        [0.9, 0.95, 0.99], 
        [0.9, 0.95, 0.99], 
        [0.9, 0.95, 0.99], 
        [0.9, 0.95, 0.99], 
        [0.9, 0.95, 0.99], 
        [0.9, 0.95, 0.99], 
        [0.9, 0.95, 0.99], 
        [0.92, 0.95, 0.99], 
        [0.9, 0.95, 0.99], 
        [0.92, 0.95, 0.99], 
        [0.9, 0.95, 0.99], 
        [0.92, 0.95, 0.99], 
        [0.9, 0.95, 0.99], 
        [0.92, 0.95, 0.99], 
        [0.9, 0.95, 0.99], 
        [0.9, 0.95, 0.99], 
        [0.9, 0.95, 0.99], 
    ]                                  
    if ckp_path is None:
        model = SalesRadar(        
        rtm_num = 3,
        sub_rtm_num = 9,
        new_pos_type_num = 12,
        state_num = 32,
        city_group_num = 7,
        district_group_num = 16,
        lob_num = 5,
        event_tier_num = 7,
        event_cate_num = 5,
        quantiles = quantiles,
        )
        if torch.cuda.is_available(): model = nn.DataParallel(model, device_ids=[0]).to(DEVICE)
        # model = torch.compile(model)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, fused=True if torch.cuda.is_available() else False)
        it = 0
        print(f'Create model from scratch', flush=True)
    else:
        model, optimizer, it = load_ckp(ckp_path)
        print(f'Load model from {ckp_path} done, {it=}', flush=True)

    print(f'number of parameter is {sum(p.nelement() for p in model.parameters())}', flush=True)
    
    print('Start training...', flush=True)
    model = model.float()
    model.train()
    
    samples = []
    for rtm3, lob in rtm3_lobs:
        train_df_rtm3 = train_df[(train_df[index_column]==rtm3) & (train_df.fph1 == lob)]
        eval_df_rtm3 = eval_df[(eval_df[index_column]==rtm3) & (eval_df.fph1 == lob)]
        
        print(f'{rtm3=}, {lob=}, train_size={len(train_df_rtm3)}, eval_size={len(eval_df_rtm3)}', flush=True)
            
        train_loader = DataLoader(SODataset(train_df_rtm3), batch_size=BATCH_SIZE, shuffle=True)
        eval_loader = DataLoader(SODataset(eval_df_rtm3), batch_size=BATCH_SIZE*2, shuffle=True)
        samples.append((train_loader, eval_loader))
    

    from datetime import datetime
    start_t = datetime.now()
    
    for it in range(it, F_STEP): ## F_STEP batches
        
        rtm3_idx = rtm3_lobs[it%len(rtm3_lobs)][0]
        lob_idx = rtm3_lobs[it%len(rtm3_lobs)][1]
        rtm3_lob_idx = it%len(rtm3_lobs)
        train_loader, _ = samples[rtm3_lob_idx]
        train_losses, _, _ = traces[rtm3_lob_idx]

        x, x_ctx, y = next(fetch_data(train_loader))
        _, train_loss = model([rtm3_idx, lob_idx], x, x_ctx, y)

        lr = calc_lr(it)
        for param_group in optimizer.param_groups: param_group['lr'] = lr
        
        optimizer.zero_grad(set_to_none=True)
        train_loss = train_loss.mean()
        train_loss.backward()
        optimizer.step()

        train_losses.append(train_loss.detach().cpu().numpy())

        if (it+1)%200 == 0:
            for vali_idx, (_, eval_loader) in enumerate(samples):
                e_s = next(fetch_data(eval_loader))
                vali_loss, q90 = estimate_loss(model, e_s, [rtm3_lobs[vali_idx][0], rtm3_lobs[vali_idx][1]])
                traces[vali_idx][1].append(vali_loss)
                traces[vali_idx][2].append(q90)
            train_mins = (datetime.now()-start_t).total_seconds()/60
            print(f'step={it+1}, lr={lr:.6f}, train_time={train_mins:.2f} mins, train_loss:{print_trace(traces, 0)}; vali_loss:{print_trace(traces, 1)}; q90:{print_trace(traces, 2)}', flush=True)
            # print(f'step={it+1}, lr={lr:.6f}, train_time={train_mins:.2f} mins, train_loss:{print_trace(traces, 0)}; vali_loss:{print_trace(traces, 1)}', flush=True)

 
        if (it+1)%2000 == 0:
            model_path = f'model/sales_radar_{it+1}.pt'
            # model_path = os.path.join(bolt.ARTIFACT_DIR, model_path)
            print(f'Save model started..., {model_path=}', flush=True)
            raw_model = model.module if torch.cuda.is_available() else model
            torch.save({
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iter_num': it+1,
            }, model_path)
            print(f'Save model done...', flush=True)

    model.eval()
    print(f'Training done, training loss is {train_loss:.4f}', flush=True)

def load_samples(train_path, eval_path):
    train_df = pd.read_csv(train_path)
    eval_df = pd.read_csv(eval_path)
    # train_df = train_df[train_df.iv > 0]
    # eval_df = eval_df[eval_df.iv > 0]    
    data_cols = [
        'rtm',
        'sub_rtm',
        'province',
        'city_group',
        'district_group',
        'fph1',
        'event_tier',
        'event_category',
        'days_since_latest_model_npi',
        'new_pos_type',        
        'so_city_group_avg_1d',
        'so_city_group_avg_3d',
        'so_city_group_avg_7d',
        'so_city_group_avg_14d',
        'so_city_group_avg_30d',
        'duration_city_group_avg_1d',
        'duration_city_group_avg_3d',
        'duration_city_group_avg_7d',
        'duration_city_group_avg_14d',
        'duration_city_group_avg_30d',
        'so_province_avg_1d',
        'so_province_avg_3d',
        'so_province_avg_7d',
        'so_province_avg_14d',
        'so_province_avg_30d',
        'duration_province_avg_1d',
        'duration_province_avg_3d',
        'duration_province_avg_7d',
        'duration_province_avg_14d',
        'duration_province_avg_30d',
        'so_district_group_avg_1d',
        'so_district_group_avg_3d',
        'so_district_group_avg_7d',
        'so_district_group_avg_14d',
        'so_district_group_avg_30d',
        'duration_district_group_avg_1d',
        'duration_district_group_avg_3d',
        'duration_district_group_avg_7d',
        'duration_district_group_avg_14d',
        'duration_district_group_avg_30d',
        'so_national_avg_1d',
        'so_national_avg_3d',
        'so_national_avg_7d',
        'so_national_avg_14d',
        'so_national_avg_30d',
        'duration_national_avg_1d',
        'duration_national_avg_3d',
        'duration_national_avg_7d',
        'duration_national_avg_14d',
        'duration_national_avg_30d',
        'traffic_cnt_pos_avg_1d',
        'traffic_cnt_pos_avg_3d',
        'traffic_cnt_pos_avg_7d',
        'traffic_cnt_pos_avg_14d',
        'traffic_cnt_pos_avg_30d',             
        index_column,
        'duration' 
    ]
    train_df = train_df[data_cols]
    eval_df = eval_df[data_cols]
    print(f'{train_path=}, {eval_path=}', flush=True)
    print(f'{len(train_df)=}, {len(eval_df)=}', flush=True)
    return (train_df, eval_df)

if __name__ == '__main__':
    print(sys.argv)

    train_df, eval_df = load_samples('data/train_data.csv', 'data/val_data.csv')
    ckp = None
    
    train(train_df, eval_df, ckp_path=ckp)
