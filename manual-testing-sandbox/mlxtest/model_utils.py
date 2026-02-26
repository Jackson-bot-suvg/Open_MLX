# import pytorch_lightning as pl
import torch
from torch import nn

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'
DROPOUT = 0.2
N_TIME_LAYER = 6
N_FEAT_LAYER = 6

class RevIN(nn.Module):
    def __init__(self, num_feats, eps=1e-5, affine=True):
        """
        :param num_feats: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_feats = num_feats
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_feats))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_feats))

    def forward(self, x, mode:str, idx=[0]):
        if mode == 'norm':
            self._build_stats(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x, idx)
        else: raise NotImplementedError
        return x

    def _build_stats(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.std = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.std
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x, idx):
        if self.affine:
            x = x - self.affine_bias[idx]
            x = x / (self.affine_weight[idx] + self.eps*self.eps)
        x = x * self.std[:, 0, idx]
        x = x + self.mean[:, 0, idx]
        return x

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        ret = 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
        return ret

class Log1p_Exp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.where(x<20, torch.log1p(torch.exp(x)), x)


class ResSin(nn.Module):
    """
    Implementation of paper 'Neural Networks Fail to Learn Periodic Functions and How to Fix It' from NIPS 2020
    """
    def forward(self, x):
        ret = x + torch.sin(x)**2
        return ret

class MXBlock(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.ln = nn.LayerNorm(n_embd)
        self.fw = nn.Linear(n_embd, n_embd)
        self.actv = ResSin()
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        x = self.ln(x)
        x = self.fw(x)
        x = self.actv(x)
        x = self.dropout(x)
        return x

class TimeMixer(nn.Module):
    def __init__(self, time_step, n_embd):
        super().__init__()
        self.inp = nn.Linear(time_step, n_embd)
        self.time_mixer = nn.Sequential(*[MXBlock(n_embd) for _ in range(N_TIME_LAYER)])
        self.out = nn.Linear(n_embd, time_step)

    def forward(self, x):
        x = self.inp(x)
        x = self.time_mixer(x)
        x = self.out(x)
        return x

class FeatureMixer(nn.Module):
    def __init__(self, chan_in, n_embd):
        super().__init__()
        self.inp = nn.Linear(chan_in, n_embd)
        self.feat_mixer = nn.Sequential(*[MXBlock(n_embd) for _ in range(N_FEAT_LAYER)])

    def forward(self, x, emb):
        # x = self.inp(x)
        if emb is None:
            x = self.feat_mixer(x)
        else:
            x = self.feat_mixer(x+emb)
        return x

class MixerBlock(nn.Module):
    def __init__(self, time_step, chan_dim, n_embd):
        super().__init__()
        self.time_mixer = TimeMixer(time_step, n_embd)
        self.feat_mixer = FeatureMixer(chan_dim, n_embd)

    def forward(self, x, emb=None):
        time_x = torch.permute(x, (0, 2, 1))  # batch_size, channel, time_step
        time_x = self.time_mixer(time_x)
        time_x = torch.permute(time_x, (0, 2, 1))  # batch_size, time_step, channel

        feat_x = time_x + x
        feat_x = self.feat_mixer(feat_x, emb)  # batch_size, time_step, channel        
        return feat_x


class SharedSpecBlock(nn.Module):
    
    def __init__(self, time_step, n_channel, n_embd, n_shared=4, n_spec=4, n_attrs=[4, 1]):

        super().__init__()

        self.shared_sa_exps = nn.ModuleList([MixerBlock(time_step, n_channel, n_embd) for _ in range(n_shared)])
        self.shared_ln = nn.ModuleList([nn.LayerNorm(n_embd) for _ in range(n_shared)])
        self.shared_gate = ModTopKGate(n_shared, n_shared, n_attrs)
        
        self.spec_sa_exps = nn.ModuleList([MixerBlock(time_step, n_channel, n_embd) for _ in range(n_spec)])
        self.spec_ln = nn.ModuleList([nn.LayerNorm(n_embd) for _ in range(n_spec)])
        self.spec_gate = ModTopKGate(n_spec, 1, n_attrs)

    def forward(self, inputs, emb = None):
        x_e, emb = inputs  # 解包元组参数
        x, attr = x_e
        shared_wgt, _ = self.shared_gate(x, attr)
        spec_wgt, _ = self.spec_gate(x, attr)   

        shared_actv_exps = torch.nonzero(shared_wgt > 0, as_tuple=False).squeeze(-1)
        spec_actv_exps = torch.nonzero(spec_wgt > 0, as_tuple=False).squeeze(-1)

        x_shared = x + sum([shared_wgt[i]*self.shared_sa_exps[i](self.shared_ln[i](x), emb) for i in shared_actv_exps])
        
        x_spec = x + sum([spec_wgt[i]*self.spec_sa_exps[i](self.spec_ln[i](x), emb) for i in spec_actv_exps])
        return ((x_shared+x_spec, attr), emb)


class ModTopKGate(nn.Module):
    """
    Implements the 'mod' routing strategy.
    """

    def __init__(self, n_experts, k, n_attributes=[4, 1]):
        super().__init__()
        self.n_experts = n_experts
        self.k = k
        self.n_attributes = n_attributes

    def forward(self, x, attrs):
        if self.k >= self.n_experts:
            combine_weights = [1.0 / self.n_experts] * self.n_experts
            return torch.tensor(combine_weights, device=x.device), torch.tensor(0.0, device=x.device)

        combined_idx = 0
        for i, attr in enumerate(attrs):
            multiplier = 1
            for j in range(i + 1, len(attrs)):
                multiplier *= self.n_attributes[j]
            combined_idx += attr * multiplier

        pdt_attr = 1
        for n_attr in self.n_attributes: pdt_attr *= n_attr
        step = int(self.n_experts/pdt_attr)
        if step <= 0: step  = 1
        combined_idx = combined_idx * step
        
        selected_experts = [
            (combined_idx + i) % self.n_experts for i in range(self.k)
        ]

        combine_weights = torch.zeros(self.n_experts, device=x.device)
        for expert_idx in selected_experts:
            combine_weights[expert_idx] = 1.0 / float(self.k)

        l_aux = torch.tensor(0.0, device=x.device)

        return combine_weights, l_aux

class Mixer(nn.Module):
    def __init__(self, time_step, chan_dim, n_embd, yh_size):
        super().__init__()
        self.time_mixer = TimeMixer(time_step, n_embd)
        self.feat_mixer = FeatureMixer(chan_dim, n_embd)
        self.fcst_head = nn.Linear(n_embd, yh_size)

    def forward(self, x, emb=None):
        time_x = torch.permute(x, (0, 2, 1))  # batch_size, channel, time_step
        time_x = self.time_mixer(time_x)
        time_x = torch.permute(time_x, (0, 2, 1))  # batch_size, time_step, channel

        feat_x = time_x + x
        feat_x = self.feat_mixer(feat_x, emb)  # batch_size, time_step, channel
        return self.fcst_head(feat_x[:, -1, :])
    
if __name__ == '__main__':
    mix = Mixer(56, 192, 192, 3)
    # mix= nn.DataParallel(mix, device_ids=[0,1]).to(DEVICE)
    print(f'number of parameter is {sum(p.nelement() for p in mix.parameters())}')
    print(f'{DEVICE=}')

    x = torch.randn([10, 56, 192], dtype=torch.float32)
    x = mix(x)