import torch
import torch.nn as nn
import math
# from helpers import to_2tuple
import torch.nn.functional as F

# From PyTorch internals
from itertools import repeat
import collections.abc

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < round_limit * v:
        new_v += divisor
    return new_v

class DynamicGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, num_nodes):
        super(DynamicGraphConvolution, self).__init__()

        self.static_adj = nn.Sequential(
            nn.Conv1d(num_nodes, num_nodes, 1, bias=False),
            nn.LeakyReLU(0.2))
        self.static_weight = nn.Sequential(
            nn.Conv1d(in_features, out_features, 1),
            nn.LeakyReLU(0.2))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv_global = nn.Conv1d(in_features, in_features, 1)
        self.bn_global = nn.BatchNorm1d(in_features)
        self.relu = nn.LeakyReLU(0.2)

        self.conv_create_co_mat = nn.Conv1d(in_features * 2, num_nodes, 1)
        self.dynamic_weight = nn.Conv1d(in_features, out_features, 1)

    def forward_static_gcn(self, x):
        x = self.static_adj(x.transpose(1, 2))
        x = self.static_weight(x.transpose(1, 2))
        return x

    def forward_construct_dynamic_graph(self, x):
        x_glb = self.gap(x)
        x_glb = self.conv_global(x_glb)
        x_glb = self.bn_global(x_glb)
        x_glb = self.relu(x_glb)
        x_glb = x_glb.expand(x_glb.size(0), x_glb.size(1), x.size(2))

        x = torch.cat((x_glb, x), dim=1)
        dynamic_adj = self.conv_create_co_mat(x)
        dynamic_adj = torch.sigmoid(dynamic_adj)
        return dynamic_adj

    def forward_dynamic_gcn(self, x, dynamic_adj):
        x = torch.matmul(x, dynamic_adj)
        x = self.relu(x)
        x = self.dynamic_weight(x)
        x = self.relu(x)
        return x

    def forward(self, x):
        out_static = self.forward_static_gcn(x)
        x = x + out_static
        dynamic_adj = self.forward_construct_dynamic_graph(x)
        finnal = self.forward_dynamic_gcn(x, dynamic_adj)
        return finnal

class ADD_GCN(nn.Module):
    def __init__(self, dim, num_classes, act_layer=nn.GELU, drop=0.):
        super(ADD_GCN, self).__init__()
        self.num_classes = num_classes
        self.fc = nn.Conv1d(dim, num_classes, 1, bias=False)
        self.conv_transform = nn.Conv1d(512, 512, 1)
        self.relu = nn.LeakyReLU(0.2)
        self.gcn = DynamicGraphConvolution(512, 512, num_classes)
        self.mask_mat = nn.Parameter(torch.eye(self.num_classes).float())
        self.last_linear = nn.Conv1d(512, self.num_classes, 1)
        self.ln0 = nn.LayerNorm(dim)
        self.ln1 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * 4)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.dropout = nn.Dropout(0.)

        self.text_head = TextHead(dim, num_classes)



    def forward_sam(self, x):

        mask = self.fc(x)
        mask = torch.sigmoid(mask)
        mask = mask.transpose(1, 2)

        x = self.conv_transform(x)
        x = torch.matmul(x, mask)
        return x


    def forward_dgcn(self, x):
        x = self.gcn(x)
        return x

    def forward(self, x):

        v = self.forward_sam(x)
        z = self.forward_dgcn(v)
        z = v + z
        out = self.ln0(z.transpose(-2, -1))
        out = out + self.dropout(self.mlp(out))
        out = self.ln1(out)
        logits = self.text_head(out)

        return logits

    def get_config_optim(self, lr, lrp):
        small_lr_layers = list(map(id, self.features.parameters()))
        large_lr_layers = filter(lambda p: id(p) not in small_lr_layers, self.parameters())
        return [
            {'params': self.features.parameters(), 'lr': lr * lrp},
            {'params': large_lr_layers, 'lr': lr},
        ]

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TextHead(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = torch.sum(x * self.weight, 2)
        if self.bias is not None:
            x = x + self.bias
        return x

