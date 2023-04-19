import numpy as np
import torch
import torch.nn.functional as F

from .structure_pruning import \
    CONV, FC, BN, GN, CONCAT, NON_PASS, PASS, NORM
EP=1e-6

def get_metric(metric_name):
    """ Get metric function by name.
    Args:
        metric_name: str, name of metric.
    Return:
        metric: function, metric function.
    """
    if metric_name == 'L0':
        metric = L0Norm
    elif metric_name == 'L1':
        metric = L1Norm
    elif metric_name == 'L2':
        metric = L2Norm
    elif metric_name == 'Linf':
        metric = LinfNorm
    elif metric_name == 'FPGM':
        metric = FPGM
    elif metric_name == 'APOZ':
        metric = APOZ
    elif metric_name == 'Hrank':
        metric = Hrank
    elif metric_name == 'CDP':
        metric = CDP
    else:
        raise NotImplementedError
    return metric
def constrain(parameter:torch.Tensor, mask=None):
    """
        Used to constrain shapes of para and mask.
        Mask should have same shape with parameter.
    """
    assert len(parameter.shape)==4
    if mask: 
        assert parameter.shape==mask.shape
        para=torch.mul(parameter,mask)
    else:
        para=parameter
    return parameter

""" Metric for conv layer parameter. The Bigger, the better.
    Args:
        parameter: tensor.shape(C_out,C_int,W,W)
        mask:tensor.shape(C_out,C_int,W,W)
            In case some C_in channels need to be set to zeros.
    Return:
        pind: the pruning metric of parameters.
"""
# torch.linalg has functions to get norm
def L0Norm(parameter:torch.Tensor, mask=None,standardize=False,**kwargs):
    para=constrain(parameter,mask)
    pind=para.gt(EP).sum((1,2,3))
    return pind

def L1Norm(parameter:torch.Tensor, mask=None,standardize=False,**kwargs):
    para=constrain(parameter,mask)
    pind=para.abs().sum((1,2,3))
    return pind

def L2Norm(parameter:torch.Tensor, mask=None,standardize=False,**kwargs):
    para=constrain(parameter,mask)
    pind=para.square().sum((1,2,3)).sqrt()
    return pind
    
def LinfNorm(parameter:torch.Tensor, mask=None,standardize=False,**kwargs):
    para=constrain(parameter,mask)
    pind=para.view(para.size(0),-1).abs().max(-1)[0]
    return pind

# Features-based
def FPGM(parameter:torch.Tensor, mask=None,**kwargs):
    """ C_out,C_in,W,W """
    para=constrain(parameter,mask)
    fpgm_list=[]
    for i in range(para.size(0)):
        d2 = (para-para[i]).square()
        d_sqrt=d2.sum(dim=(1,2,3)).sqrt()
        fpgm_list.append(torch.mean(d_sqrt))
    pind=torch.tensor(fpgm_list)
    return pind

def APOZ(features:torch.Tensor, mask=None,**kwargs):
    """ N,C,H,W """
    feas=constrain(features,mask)
    e=feas.mean()/100
    normd_output=F.relu(feas-feas.mean(dim=(0,2,3),keepdim=True))
    pind=1-torch.true_divide(
        (normd_output.abs().le(e)).int().sum(dim=(0,2,3)),
        feas.size(0)*feas.size(2)*feas.size(3)
    )
    return pind

def Hrank(features:torch.Tensor, mask=None,**kwargs):
    """N,C,H,W"""
    feas=constrain(features,mask)
    normd_output=F.relu(feas-feas.mean(dim=(0,2,3),keepdim=True))
    a,b,_,_=normd_output.size()
    rank=torch.Tensor(a,b)
    for i in range(a):
        for j in range(b):
            rank[i][j]=torch.matrix_rank(normd_output[i,j,:,:]).item()
    pind=rank.sum(dim=0)/normd_output.size(0)
    return pind

def feature_pool(features:torch.Tensor):
    features=F.relu(features).mean(dim=(2,3)).transpose(0,1)
    return features

def CDP(features:torch.Tensor, mask=None, coe:int=20,**kwargs):
    """ N,C,H,W """
    if len(features.shape)==4:
        features=constrain(features,mask)
        features=feature_pool(features)
    features=features.cpu()
    E=torch.exp(torch.ones(1)).cpu()
    balance_coe = torch.log((features.size(0)/coe)*E) if coe else torch.ones(1)
    sample_mean = features.mean(dim=1).view(features.size(0),1) # 计算每个通道各自的激活图均值
    over_ave=(features>=sample_mean).int() # [64,10000] 每个通道下10k数据里有多少个超过均值的
    #calc tf & tf mean
    tf=(features/features.sum(dim=0))*balance_coe # [64,10000]/[1,10000]*[1]=[64,10000] 通道维度上的归一化
    tf_mean=(tf*over_ave).sum(dim=1)/over_ave.sum(dim=1) # [64,10000]*[64,10000]=>[64] / [64,10000]=>[64] =[64] 通道维度上的加权平均
    #calc idf
    idf=torch.log(features.size(1)/(over_ave.sum(dim=1)+1.0)) # [1]/[64,10000]=>[64] log((样本总数)/(该通道下超过均值的样本个数+1))) 

    importance=tf_mean*idf
    return importance

WEIGHT_METRICS={
    "L0":L0Norm,
    "L1":L1Norm,
    "L2":L2Norm,
    "Linf":LinfNorm,
    "FPGM":FPGM,
}
FILTER_METRICS={
    "HRANK":Hrank,
    "APOZ":APOZ,
    "CDP":CDP,
}
METRICS={**WEIGHT_METRICS,**FILTER_METRICS}