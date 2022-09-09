# Copyright (c) OpenMMLab. All rights reserved.
import copy

import mmcv
import torch
import torch.nn as nn
from mmcv.cnn import get_model_complexity_info
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models.builder import ALGORITHMS, build_pruner
from mmrazor.models.utils import add_prefix
from .base import BaseAlgorithm

@ALGORITHMS.register_module()
class CommonPruningAlgoritm(BaseAlgorithm):
    def __init__(self, input_shape=(3,224,224),**kwargs):
        super().__init__(**kwargs)
        if input_shape is not None:
            self.input_shape = input_shape
            self._init_flops()
    # from autoslim
    def _init_pruner(self, pruner):
        """Build registered pruners and make preparations.

        Args:
            pruner (dict): The registered pruner to be used
                in the algorithm.
        """
        if pruner is None:
            self.pruner = None
            return

        # judge whether our StructurePruner can prune the architecture
        try:
            pseudo_pruner = build_pruner(pruner)
            pseudo_architecture = copy.deepcopy(self.architecture)
            pseudo_pruner.prepare_from_supernet(pseudo_architecture)
            subnet_dict = pseudo_pruner.sample_subnet()
            pseudo_pruner.set_subnet(subnet_dict)
            subnet_dict = pseudo_pruner.export_subnet()

            pseudo_pruner.deploy_subnet(pseudo_architecture, subnet_dict)
            pseudo_img = torch.randn(1, 3, 224, 224)
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
    def get_raw_flops(self):
        flop=torch.tensor(0,dtype=torch.float64)
        for name,module in self.architecture.named_modules():
            if hasattr(module,'__flops__'):
                flop+=module.__flops__
        return flop
    
    def compress(self, ratio):
        d=self.pruner.sample_subnet_ratio_uniform(ratio)
        self.pruner.set_subnet(d)
    
    def get_flops(self):
        pass

