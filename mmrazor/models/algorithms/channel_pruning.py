# Copyright (c) OpenMMLab. All rights reserved.
import copy

import mmcv
import torch
import torch.nn as nn
from mmcv.cnn import get_model_complexity_info
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models.builder import ALGORITHMS, build_pruner
from mmrazor.models.utils import add_prefix
from mmrazor.models.pruners.metrics import get_metric, WEIGHT_METRICS, FILTER_METRICS

from .base import BaseAlgorithm
from .utils import extract_features

@ALGORITHMS.register_module()
class ChannelPruningAlgoritm(BaseAlgorithm):
    def __init__(self, input_shape=(3,224,224),**kwargs):
        self.input_shape = input_shape
        # self.metric=kwargs.get('metric',None)
        super().__init__(**kwargs)
        if input_shape is not None:
            self._init_flops()
    # from autoslim
    def _init_pruner(self, pruner):
        """Build registered pruners and make preparations.

        Args:
            pruner (dict): The registered pruner to be used
                in the algorithm.
        """
        assert isinstance(pruner, dict), 'pruner must be a dict of configs'

        # judge whether our StructurePruner can prune the architecture
        try:
            pseudo_pruner = build_pruner(pruner)
            pseudo_architecture = copy.deepcopy(self.architecture)
            pseudo_pruner.prepare_from_supernet(pseudo_architecture)
            subnet_dict = pseudo_pruner.sample_subnet()
            pseudo_pruner.set_subnet(subnet_dict)
            subnet_dict = pseudo_pruner.export_subnet()

            pseudo_pruner.deploy_subnet(pseudo_architecture, subnet_dict)
            pseudo_img = torch.randn(self.input_shape)
            pseudo_architecture.forward_dummy(pseudo_img)
        except RuntimeError:
            raise NotImplementedError('Our current StructurePruner does not '
                                      'support pruning this architecture. '
                                      'StructurePruner is not perfect enough '
                                      'to handle all the corner cases. We will'
                                      ' appreciate it if you create a issue.')

        self.pruner = build_pruner(pruner)

        if self.retraining:
            if isinstance(self.channel_cfg, dict):
                self.pruner.deploy_subnet(self.architecture, self.channel_cfg)
                self.deployed = True
            elif isinstance(self.channel_cfg, (list, tuple)):

                self.pruner.convert_switchable_bn(self.architecture,
                                                  len(self.channel_cfg))
                self.pruner.prepare_from_supernet(self.architecture)
            else:
                raise NotImplementedError
        else:
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
    
    def compress(self, reduction_ratio,**kwargs):
        self.pruner.remove_denoted_group(['downsample'])
        ratio=1-reduction_ratio
        dataloader =  kwargs.get("dataloader",None)
        metric_name=self.pruner.pruning_metric_name
        if WEIGHT_METRICS.__contains__(metric_name):
            contents= {name:getattr(module,'weight') for name,module in self.pruner.name2module.items()}
        elif FILTER_METRICS.__contains__(metric_name):
            space2name,features_dict, features_pool = extract_features(self,dataloader,True)
            contents=features_dict
        else:
            raise NotImplementedError
        metric_scores,_=self.pruner.get_metric_local(metric_name,contents)
        print("METRIC SCORES:")
        # print(metric_scores)
        for name,score in metric_scores.items():
            print(f"{name}:{score.shape}")
        subdict=self.pruner.sample_subnet_diff(ratio,metric_scores)
        self.pruner.set_subnet(subdict)
        # if kwargs.get("autoencoder",False):
        #     self.autoencoder_new_weight()
    def get_flops(self):
        pass
    
    def autoencoder_new_weight(self):
        # Generate new weights from the current weights
        from mmrazor.models.pruners.utils import ae_distil
        for i,space in enumerate(self.pruner.space2name):
            for name in self.pruner.space2name[space]:
                module=self.pruner.name2module[name]
                module,old_parameter=ae_distil(module,20)
    
    def neuron_merging(self):
        pass    
        # 获取卷积层和后续的bn
        
        # 得到被保留的通道

        # 基于这些信息,计算剪枝矩阵z
    def compress_channel_cfg(self,channel_cfg=None):
        ccfg=channel_cfg if channel_cfg else self.channel_cfg
        self.pruner.deploy_subnet(self.architecture, ccfg)
        self.deployed = True
        pass
    def compress_space2ratio(self,space2ratio=None):
        subnet_dict=self.pruner.sample_subnet_nonuni(space2ratio)
        self.pruner.set_subnet(subnet_dict)
        pass

    def test_score_compress(self, reduction_ratio,**kwargs):
        import json
        import os
        remove_subdict=self.pruner.remove_denoted_group(['downsample'])
        ratio=1-reduction_ratio
        dataloader =  kwargs.get("dataloader",None)
        metric_name=self.pruner.pruning_metric_name
        if WEIGHT_METRICS.__contains__(metric_name):
            contents= {name:getattr(module,'weight') for name,module in self.pruner.name2module.items()}
        elif FILTER_METRICS.__contains__(metric_name):
            space2name,features_dict, features_pool = extract_features(self,dataloader,True)
            contents=features_dict
        else:
            raise NotImplementedError
        metric_scores,layer2score=self.pruner.get_metric_local(metric_name,contents)
        print("METRIC SCORES:")
        # print(metric_scores)
        for name,score in metric_scores.items():
            print(f"{name}:{score.shape}")
        
        subdict=self.pruner.sample_subnet_diff(ratio,metric_scores)
        # for i,key in enumerate(remove_subdict):
        #     print(f"{i}:\t {key}:\t {remove_subdict[key].shape}:\t {remove_subdict[key].sum()}")

        subdict.update(remove_subdict)
        # extend dict
        

        self.pruner.set_subnet(subdict)

        # print number of filters in self.architecture after pruning
        for name, module in self.architecture.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                
                print(f"Layer {name} * \t{module.out_mask.shape[1]} * {int(module.out_mask.sum())}")
        
        # for layer, score in layer2score.items():
        #     layer2score[layer]=score.tolist()
        # abspath="/home/wyh/wuyuhang/mmrazor/tests/simple_pruning/CDP/visual"
        # # model_name=kwargs.get("save_compress_algo","").lstrip("./").split("_")[0]
        # model_name=kwargs.get("save_compress_algo","")
        # model_name=model_name.split("/")[-1]
        # print(f"MODEL: {model_name}")
        # file_path=os.path.join(abspath,f"{model_name}.json")
        # with open(file_path,"w") as f:
        #     json.dump(layer2score,f)
        # if kwargs.get("autoencoder",False):
        #     self.autoencoder_new_weight()
        # load json from filepath
        # with open(file_path,"r") as f:
        #     layer2score=json.load(f)