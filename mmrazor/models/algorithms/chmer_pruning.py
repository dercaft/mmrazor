# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from scipy.spatial.distance import cdist
import scipy
from geomloss import SamplesLoss

import copy
import numpy as np
import mmcv
import torch
import torch.nn as nn
import torch.functional as F
from mmcv.cnn import get_model_complexity_info
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models.builder import ALGORITHMS, build_pruner
from mmrazor.models.utils import add_prefix
from mmrazor.models.pruners import feature_pool 
from mmrazor.models.pruners.utils import mix_distribution as md
from .base import BaseAlgorithm

# from mmrazor.metrics 
# 对应的Script为: mmrazor/tests/simple_ChMer/run_chmer_pipline.sh
# 对应的Config为: mmrazor/tests/simple_ChMer/configs/chmer_pruning.py
# 对应的Main Python为: mmrazor/tests/simple_ChMer/test_mmcls.py
# 对应的Algorithm为: mmrazor/mmrazor/models/algorithms/chmer_pruning.py
# 对应的Pruner为: mmrazor/mmrazor/models/pruners/channel_pruning.py
@ALGORITHMS.register_module()
class ChannelMergingAlgoritm(BaseAlgorithm):
    def __init__(self, input_shape=(3,224,224),**kwargs):
        super().__init__(**kwargs)
        if input_shape is not None:
            self.input_shape = input_shape
            self._init_flops()
    def _init_pruner(self, pruner):
        self.pruner = build_pruner(pruner)
        self.pruner.prepare_from_supernet(self.architecture)


    def _init_flops(self):
        """Get flops information of the supernet."""
        flops_model = copy.deepcopy(self.architecture)
        flops_model.eval()
        if hasattr(flops_model, 'forward_dummy'):
            flops_model.forward = flops_model.forward_dummy
        else:
            raise NotImplementedError(
                'FLOPs counter is currently not currently supported with {}'.
                format(flops_model.__class__.__name__))

        flops, params = get_model_complexity_info(
            flops_model, self.input_shape, print_per_layer_stat=False)
        flops_lookup = dict()
        for name, module in flops_model.named_modules():
            flops = getattr(module, '__flops__', 0)
            flops_lookup[name] = flops
        del (flops_model)

        for name, module in self.architecture.named_modules():
            module.__flops__ = flops_lookup[name]

    def get_subnet_flops(self):
        """A hacky way to get flops information of a subnet."""
        flops = 0
        last_out_mask_ratio = None
        for name, module in self.architecture.named_modules():
            if type(module) in [
                    nn.Conv2d, mmcv.cnn.bricks.Conv2d, nn.Linear,
                    mmcv.cnn.bricks.Linear
            ]:
                in_mask_ratio = float(module.in_mask.sum() /
                                      module.in_mask.numel())
                out_mask_ratio = float(module.out_mask.sum() /
                                       module.out_mask.numel())
                flops += module.__flops__ * in_mask_ratio * out_mask_ratio
                last_out_mask_ratio = out_mask_ratio
            elif type(module) == nn.BatchNorm2d:
                out_mask_ratio = float(module.out_mask.sum() /
                                       module.out_mask.numel())
                flops += module.__flops__ * out_mask_ratio
                last_out_mask_ratio = out_mask_ratio
            elif type(module) in [
                    nn.ReLU, nn.PReLU, nn.ELU, nn.LeakyReLU, nn.ReLU6
            ]:

                assert last_out_mask_ratio, 'An activate module can not be ' \
                                            'the first module of a network.'
                flops += module.__flops__ * last_out_mask_ratio

        return round(flops)
    # end
    def get_supnet_flops(self):
        flop=torch.tensor(0,dtype=torch.float64)
        for name,module in self.architecture.named_modules():
            if hasattr(module,'__flops__'):
                flop+=module.__flops__
        return flop
    
    def compress(self, ratio):
        d=self.pruner.sample_subnet_uniform(ratio)
        self.pruner.set_subnet(d)
    
    def get_flops(self):
        pass

    def test_fusion_init(self,**kwargs):
        for name in self.pruner.name2module:
            module=self.pruner.name2module[name]
            if(hasattr(module,'weight')):
                print("name:{} , module:{} , weight:{}".format(name,type(module),module.weight.shape))
        print("开始层剪枝")
        chdic=self.pruner.sample_subnet()
        conv_list=[]
        for ch in chdic:
            print("ch:{}".format(ch))
            print("chdic[ch]:{}".format(chdic[ch]))
            if 'conv' in ch:
                conv_list.append(ch)
        # print("BN_LIST")
        # count=0
        # for ch in chdic:
        #     if 'bn' in ch:
        #         if count==0:
        #             print(self.pruner.name2module[ch].__dict__)
        #         count+=1
        #         print(ch," ",self.pruner.name2module[ch].weight.shape," ",torch.sum(self.pruner.name2module[ch].weight).data)
                
        print("CONV_LIST")
        for conv in conv_list:
            print(conv," ",self.pruner.name2module[conv].weight.shape," ",torch.sum(self.pruner.name2module[conv].weight).data)
        # Start to prune.
        for conv in conv_list:            
            self.pruner.prune_layer(conv)
            bn=self.pruner.find_next_bn(conv)
            if bn:
                self.pruner.prune_layer(bn)
        
        for conv in conv_list:
            print(conv," ",self.pruner.name2module[conv].weight.shape," ",torch.sum(self.pruner.name2module[conv].weight).data)
        # start to restore
        for conv in conv_list:
            self.pruner.restore_layer(conv)
            bn=self.pruner.find_next_bn(conv)
            if bn:
                self.pruner.restore_layer(bn)
        print("MODULE RESTORED")
        
    def generate_hook(self,name,fdict:dict,fpool:dict|None=None):
        def output_hook(module,input,output) -> None:
            fdict[name]=output.detach()
            if isinstance(fpool,dict) :
                fpool[name]=feature_pool(output.detach())
        return output_hook

    def extract_features(self, dataloader, get_pool:bool=False):
        """
            Use hooks to get features from model's forward.
            Return: space2name: {k:[n1,...,nn]}
        """
        supernet=self.architecture
        pruner=self.pruner

        space2name={}
        for name,module in supernet.model.named_modules():
            spi=pruner.get_space_id(name)
            if not isinstance(spi,str): continue
            g=space2name.setdefault(spi,[])
            g.append(name)
            space2name[spi]=g
        
        hookers=[]
        features_dict={}
        features_pool={} if get_pool else None
        names=[n for v in space2name.values() for n in v]
        for name, module in supernet.model.named_modules():
            if name in names:
                oo=self.generate_hook(name,features_dict,features_pool)
                hooker=module.register_forward_hook(oo)
                hookers.append(hooker)
                # self.weights_dict[name]=module.weight
                # if hasattr(module,'bias'):
                    # self.bias_dict[name]=module.bias
        
        # forward a batch
        # algorithm_for_test.
        self.architecture.eval()
        for i, data_batch in enumerate(dataloader):
            if i>0:break
            self.forward(**data_batch,return_loss=False)
        for hooker in hookers:
            hooker.remove()

        return space2name,features_dict,features_pool

    def fusion_pipeline(self,**kwargs):
        '''
            Fusion Pipline of LayerFusion
        '''
        self.architecture.eval()
        reduction_ratio = kwargs.get('reduction_ratio', 1)
        reduction_flops = self.get_supnet_flops()*reduction_ratio

        # 
        dataloader=kwargs.get('dataloader', None)
        assert dataloader, 'Need valid dataloader'

        # get weight/features, using model itself or features from inference
        space2name,features_dict, features_pool = self.extract_features(dataloader,get_pool=True)
        for k in features_pool:
            # features_pool[k]=features_pool[k].T.numpy()
            features_pool[k]=features_pool[k].T
        # show keys and shapes in features_dict and feautres_pool
        
        # print("ATTENTION*******")
        # for i, fk in enumerate(features_dict):
        #     print("* {} * fk: {}, shape: {}".format(i,fk,features_dict[fk].shape))
        # print('POOL************')
        # for i, fk in enumerate(features_pool):
        #     print("* {} * fk: {}, shape: {}".format(i,fk,features_pool[fk].shape))
        
        
        # del features_dict
        # features_dict= features_pool
        # print info about space2name features_dict
        # for i,(spn,names) in enumerate(space2name.items()):
        #     print("{} SPN: {}, NAM: {}".format(i,spn,names))
        #     for j, n in enumerate(names):
        #         f=features_dict.get(n,None)
        #         # fs=f.shape if not isinstance(f, type(None)) else 0
        #         fs=f.shape
        #         print(" {} FEAT: {}, SHAPE: {}".format(j,n,fs))
        # # group module according to their feature shape
        groups_features={}
        for name,module in self.architecture.named_modules():
            if name.startswith('model.'):
                name=name[6:]
            if name not in features_dict: continue
            f=features_dict[name]
            shape=tuple(f.shape[1:])
            groups_features[shape]=groups_features.setdefault(shape,[])
            groups_features[shape].append(name)

        # group module according to their weight shape
        groups_weights={}
        for name,module in self.architecture.named_modules():
            if name.startswith('model.'):
                name=name[6:]
            if not hasattr(module,'weight') or name not in features_dict: continue
            shape=tuple(module.weight.shape)
            groups_weights[shape]=groups_weights.setdefault(shape,[])
            groups_weights[shape].append(name)
        
        # print groups_features:
        print("#### GROUPS FEATURES")
        for i,(shape,names) in enumerate(groups_features.items()):
            print("{} SHAPE: {}, NAM: {}".format(i,shape,names))
        # print group_weights:
        print("#### GROUPS WEIGHTS")
        for i,(shape,names) in enumerate(groups_weights.items()):
            print("{} SHAPE: {}, NAM: {}".format(i,shape,names))

        # calculate distance/ similarity/ importance in each group
        # data structure:
        #   groups={
        #       (shape(1,2,3)):[n1,n2,...,nn],
        #   }
        #   dis={
        #       (shape(1,2,3)):{
        #           {pair(n1,n2):dis,...,}
        #       }
        #   }
        loss=SamplesLoss(loss='sinkhorn', p=2, blur=0.5)
        dis={}
        for shape in groups_weights:
            dis[shape]=dis.setdefault(shape,{})
            for names in groups_weights[shape]:
                dis[shape]=self.cal_dis(groups_weights[shape],features_pool,loss)

        # 选取满足要求的要被剪掉的层
        # # Select Units from dis/sim/imp
        # # globally compare all pairs. Greedy selections.
        name2score=OrderedDict()
        for shape in dis:
            names=[]
            for pair in dis[shape]:
                score=dis[shape][pair]
                s=name2score.setdefault(pair[0],0)
                name2score[pair[0]]=s+score
                s=name2score.setdefault(pair[1],0)
                name2score[pair[1]]=s+score
                names+=[pair[0],pair[1]]
            names=set(names)
            # for name in groups[shape]:
            for name in names:
                s=name2score.get(name,0)
                name2score[name]=s/len(names)
            
        score2name=sorted(name2score.items(),key=lambda x:x[1],reverse=False)

        print("#### SCORE2NAME: ")
        for name in score2name:
            print(name," ")
        # select one reserved layer for each group
        reserved_layers=[]
        for shape in groups_weights:
            max_score,max_name=0,''
            for name in groups_weights[shape]:
                score=name2score.get(name,-1)
                if score>max_score:
                    max_score=score
                    max_name=name
            reserved_layers.append(max_name)
        self.pruned_layers_name=[]
        rf=0
        for i,(name,score) in enumerate(score2name):
            sn=name.replace('model.','')
            if sn in reserved_layers:
                continue
            rf+=self.pruner.name2module[sn].__flops__
            self.pruned_layers_name.append(sn)
            if rf>=reduction_flops:
                break
        print("PRUNED LAYERS: ")
        for i,pln in enumerate(self.pruned_layers_name):
            print("{}: {}".format(i,pln))
        # # execute merging operation
        # select nearst layer as aim layer to merge
        for pruned_layer in self.pruned_layers_name:
            sims=[]
            for key in groups_weights:
                if pruned_layer in groups_weights[key]:
                    for pair in dis[key]:
                        another=pair[0] if pair[1]==pruned_layer else pair[1]
                        if pruned_layer in pair and another not in self.pruned_layers_name:
                            sims.append((pair,dis[key][pair]))
                        # if pruned_layer in pair :
                        #     sims.append((pair,dis[key][pair]))
            sims=sorted(sims,key=lambda x:x[1],reverse=False)
            
            print("SIMS of {}:".format(pruned_layer))
            _=[print("{}".format(x)) for x in sims]
            
            if len(sims):
                aim_layer=sims[0][0][0] if sims[0][0][1]==pruned_layer else sims[0][0][1]
                
                # self.layer_merge(aim_layer,pruned_layer,method='mix')
                # self.layer_merge_bn(self.pruner.find_next_bn(aim_layer), \
                #                     self.pruner.find_next_bn(pruned_layer),method='mix')
                
                aim_bn=self.pruner.find_next_bn(aim_layer)
                pruned_bn=self.pruner.find_next_bn(pruned_layer)
                self.layer_merge_scale(aim_layer,pruned_layer,aim_bn,pruned_bn)
                # self.pruner.prune_layer(aim_bn)
                # print("BN: mean:{} var:{}".format(self.pruner.name2module[aim_bn].running_mean,self.pruner.name2module[aim_bn].running_var))
                self.pruner.skip_module(aim_bn)

            # else:
            #     spi=self.pruner.get_space_id(pruned_layer)

        # # execute pruning operation 会改变原模型的权重，所以要先融合权重，再屏蔽被剪枝权重
        # for layer in self.pruned_layers_name:
        #     # self.pruner.prune_layer(layer)
        #     self.pruner.skip_module(layer)
        #     bn=self.pruner.find_next_bn(layer)
        #     if bn:
        #         self.pruner.skip_module(bn)
                # self.pruner.prune_layer(bn)
        
        # for layer in self.pruned_layers_name:
        #     self.compare_features(dataloader=dataloader,aim_layer=aim_layer,pruned_layer=layer)            
    
    def cal_dis(self,names,features_dict,loss=None):
        '''
            Calculate distance between units in a group.
        '''
        # calculate distance between units
        dis={}
        for i in range(len(names)):
            for j in range(i+1,len(names)):
                n1,n2=names[i],names[j]
                f1,f2=features_dict[n1],features_dict[n2]
                if i==0 and j==1:
                    print("TYPE AND SHAPE: ",type(f1),f1.shape)
                # dis[(n1,n2)]=cdist(f1, f2, metric='euclidean', p=2)
                dis[(n1,n2)]=float(loss(f1,f2))
                # scipy.stats.wasserstein_distance(f1,f2)
                # dis[(n1,n2)]=scipy.stats.wasserstein_distance(f1,f2)
        return dis
    
    def layer_merge(self,aim_layer,pruned_layer,method:str='mix'):
        '''
            Merge two layers into one layer. Only change aim layer.
            layer_name_0: the name of the aim layer
            layer_name_1: the name of the pruned layer
        '''
        assert self.pruner.name2module.get(aim_layer,None) is not None
        assert self.pruner.name2module.get(pruned_layer,None) is not None
        assert method in ['mix','mean','max','expand'] # 大于mean的选max，小于mean的选min
        w0=self.pruner.name2module[aim_layer].weight.data
        w1=self.pruner.name2module[pruned_layer].weight.data
        # b0=self.pruner.name2module[aim_layer].bias
        # b1=self.pruner.name2module[pruned_layer].bias
        if method == 'mean':
            self.pruner.name2module[aim_layer].weight = torch.nn.Parameter((w0+w1)/2)
            # self.pruner.name2module[aim_layer].bias = torch.nn.Parameter((b0+b1)/2)
        elif method == 'max':
            self.pruner.name2module[aim_layer].weight = torch.nn.Parameter(torch.max(w0,w1))
            # self.pruner.name2module[aim_layer].bias = torch.nn.Parameter(torch.max(b0,b1))
        elif method == 'mix':
            self.pruner.name2module[aim_layer].weight = torch.nn.Parameter(md.random_mix(w1,w0,p=0.5,disc_rep=True, eps=0, scale=True))
            # self.pruner.name2module[aim_layer].bias = torch.nn.Parameter(md.random_mix(b1,b0,p=0.3,disc_rep=False, eps=0, scale=True))
        elif method == 'expand':
            self.pruner.name2module[aim_layer].weight = torch.nn.Parameter(md.expand_dist(w1,w0))
            # self.pruner.name2module[aim_layer].bias = torch.nn.Parameter(md.expand_dist(b1,b0))
    
    def layer_merge_bn(self,aim_layer,pruned_layer,method:str='mix'):
        '''
            Merge two layers into one layer. Only change aim layer.
            layer_name_0: the name of the aim layer
            layer_name_1: the name of the pruned layer
        '''
        possibile=0.5
        disc_rep=True
        scale=True
        assert self.pruner.name2module.get(aim_layer,None) is not None, "aim_layer: {} not found".format(aim_layer)
        assert self.pruner.name2module.get(pruned_layer,None) is not None, "pruned_layer: {} not found".format(pruned_layer)
        bn0=self.pruner.name2module[aim_layer]
        bn1=self.pruner.name2module[pruned_layer]
        if method =='mean':        
            bn0.weight.data = torch.nn.Parameter((bn0.weight.data+bn1.weight.data)/2)
            bn0.bias.data = torch.nn.Parameter((bn0.bias.data+bn1.bias.data)/2)
            bn0.running_mean.data = torch.nn.Parameter((bn0.running_mean.data+bn1.running_mean.data)/2)
            bn0.running_var.data = torch.nn.Parameter((bn0.running_var.data+bn1.running_var.data)/2)
        elif method == 'mix':
            bn0.weight.data = torch.nn.Parameter(md.random_mix(bn1.weight.data,bn0.weight.data,p=possibile,disc_rep=disc_rep, eps=0, scale=scale))
            bn0.bias.data = torch.nn.Parameter(md.random_mix(bn1.bias.data,bn0.bias.data,p=possibile,disc_rep=disc_rep, eps=0, scale=scale))
            bn0.running_mean.data = torch.nn.Parameter(md.random_mix(bn1.running_mean.data,bn0.running_mean.data,p=possibile,disc_rep=disc_rep, eps=0, scale=scale))
            bn0.running_var.data = torch.nn.Parameter(md.random_mix(bn1.running_var.data,bn0.running_var.data,p=possibile,disc_rep=disc_rep, eps=0, scale=scale))
        elif method == 'expand':
            bn0.weight.data = torch.nn.Parameter(md.expand_dist(bn1.weight.data,bn0.weight.data))
            bn0.bias.data = torch.nn.Parameter(md.expand_dist(bn1.bias.data,bn0.bias.data))
            bn0.running_mean.data = torch.nn.Parameter(md.expand_dist(bn1.running_mean.data,bn0.running_mean.data))
            bn0.running_var.data = torch.nn.Parameter(md.expand_dist(bn1.running_var.data,bn0.running_var.data))

    def layer_merge_scale(self,aim_layer,pruned_layer,aim_bn,pruned_bn):
        '''
            Merge two layers into one layer. Only change aim layer, scale pruned_layer based on two bns.
            aim_layer: the name of the aim layer
            pruned_layer: the name of the pruned layer
        '''
        assert self.pruner.name2module.get(aim_layer,None) is not None
        assert self.pruner.name2module.get(pruned_layer,None) is not None
        assert self.pruner.name2module.get(aim_bn,None) is not None
        assert self.pruner.name2module.get(pruned_bn,None) is not None
        conv0=self.pruner.name2module[aim_layer]
        bn0=self.pruner.name2module[aim_bn]
        conv1=self.pruner.name2module[pruned_layer]
        bn1=self.pruner.name2module[pruned_bn]

        weight0,bias0=self.fuse_bn_into_conv(conv0,bn0)
        weight1,bias1=self.fuse_bn_into_conv(conv1,bn1)
        
        # print("weight0 shape: ",weight0.shape)
        # print("bias0 shape: ",bias0.shape)
        # print("weight1 shape: ",weight0.shape)
        # print("bias1 shape: ",bias0.shape)

        # merge bias into weight
        # weight0=weight0+bias0.view(-1,1,1,1)
        # weight1=weight1+bias1.view(-1,1,1,1)
        # merge weight1 into weight0
        # weight0=(weight0+weight1)/2
        # bias0=(bias0+bias1)/2
        # weight0 = md.random_mix(weight1,weight0,p=0.5,disc_rep=True, eps=0, scale=True)
        # bias0 = md.random_mix(bias1,bias0,p=0.5,disc_rep=True, eps=0, scale=True)
        # print shape of parameters
        
        self.pruner.name2module[aim_layer].weight = torch.nn.Parameter(weight0)
        self.pruner.name2module[aim_layer].bias = torch.nn.Parameter(bias0)

    def fuse_bn_into_conv(self,conv,bn):
        '''
            Fuse batch normalization into convolutional layer.
            conv: the convolutional layer
            bn: the batch normalization layer
        '''
        shape=conv.weight.size()
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps+bn.running_var)))
        w_conv=torch.mm(w_bn, w_conv).view(shape)
        
        # setting bias
        if conv.bias is not None:
            b_conv = conv.bias
        else:
            b_conv = torch.zeros( conv.weight.size(0) )
        b_conv = torch.mm(w_bn, b_conv.view(-1, 1)).view(-1)
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
                              torch.sqrt(bn.running_var + bn.eps)
                            )
        bias=( b_conv + b_bn )
        return w_conv,bias

    def compare_features(self,dataloader,aim_layer,pruned_layer):
        '''
            Compare features of two layers.
            aim_layer: the name of the aim layer
            pruned_layer: the name of the pruned layer
        '''
        assert self.pruner.name2module.get(aim_layer,None) is not None
        assert self.pruner.name2module.get(pruned_layer,None) is not None
        # assert self.pruner.name2module.get(aim_bn,None) is not None
        # assert self.pruner.name2module.get(pruned_bn,None) is not None
        
        space2name,features_dict,_=self.extract_features(dataloader,get_pool=False)
        # find the previous layer output
        
        before=features_dict[self.pruner.node2parents[pruned_layer][0]]
        # before=torch.clamp(before,min=0)
        after=features_dict[pruned_layer]
        print("last name: {}. now name: {}".format(self.pruner.node2parents[pruned_layer][0],pruned_layer))
        print("before shape: ",before.shape," after shape: ",after.shape)
        result=torch.all(before==after)
        print("IS SAME?: {}".format(result))