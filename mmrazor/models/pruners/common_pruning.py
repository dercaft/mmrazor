# Copyright (c) OpenMMLab. All rights reserved.
from pydoc import doc
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules import GroupNorm

from mmrazor.models.builder import PRUNERS
from .structure_pruning import StructurePruner
from .utils import SwitchableBatchNorm2d


@PRUNERS.register_module()
class CommonPruner(StructurePruner):
    """A random ratio pruner.

    Each layer can adjust its own width ratio randomly and independently.

    Args:
        ratios (list | tuple): Width ratio of each layer can be
            chosen from `ratios` randomly. The width ratio is the ratio between
            the number of reserved channels and that of all channels in a
            layer. For example, if `ratios` is [0.25, 0.5], there are 2 cases
            for us to choose from when we sample from a layer with 12 channels.
            One is sampling the very first 3 channels in this layer, another is
            sampling the very first 6 channels in this layer. Default to None.
    """

    def __init__(self, **kwargs):
        super(CommonPruner, self).__init__(**kwargs)
        self.ratios = []

    def _check_module_pruner(self,module,ratio):
        if isinstance(module,GroupNorm):
            num_channels = module.num_channels
            num_groups = module.num_groups
            new_channels = int(round(num_channels * ratio))
            assert (num_channels * ratio) % num_groups == 0, \
                f'Expected number of channels in input of GroupNorm ' \
                f'to be divisible by num_groups, but number of ' \
                f'channels may be {new_channels} according to ' \
                f'ratio {ratio} and num_groups={num_groups}'
    
    def prepare_from_supernet(self, supernet):
        super(CommonPruner, self).prepare_from_supernet(supernet)

    def get_channel_mask_randomly(self,out_mask):
        """ Randomly choose a width ratio of a layer
            out_mask: (1,channels,1,1)
        """ 
        while True:
            random_ratio=1-np.random.rand()
            if random_ratio>1/out_mask.size(1): break
        new_out_mask=self.get_channel_mask_ratio(out_mask=out_mask,ratio=random_ratio)
        return new_out_mask
    def get_channel_mask_ratio(self,out_mask,ratio,metric=None):
        """ Denote ratio and metric to create a new mask.
            out_mask: torch.Tensor shape(1,out_channels,1,1)
            ratio: float(0,1]
            metric: torch.Tensor shape(out_channels,) # Bigger, better
        """
        assert ratio>0 and ratio<=1, f'Ratio out of range, now is: {ratio}'
        if metric:
            assert isinstance(metric,torch.Tensor) and len(metric) == out_mask.size(1), \
                f'Metric length is mismatch with out_mask,' \
                f'Metric size is: {metric.size()}, out_mask size is: {out_mask.size()}'
        new_channels=int(round(out_mask.size(1)*ratio))
        assert new_channels >0, f'Output channels should be a positive integer. ratio is {ratio}.'

        new_out_mask=torch.zeros_like(out_mask)
        if metric:
            new_indx=torch.topk(metric,new_channels)[1]
            new_out_mask.index_fill_(dim=1,index=new_indx,value=1)
        else:
            new_out_mask[:,:new_channels]=1
        return new_out_mask

    def sample_subnet_randomly(self):
        """Random sample subnet by random mask.

        Returns:
            dict: Record the information to build the subnet from the supernet,
                its keys are the properties ``space_id`` in the pruner's search
                spaces, and its values are corresponding sampled out_mask.
        """
        subnet_dict = dict()
        for space_id, out_mask in self.channel_spaces.items():
            subnet_dict[space_id] = self.get_channel_mask_randomly(out_mask)
        return subnet_dict

    def sample_subnet_ratio_uniform(self,ratio):
        subnet_dict = dict()
        for space_id, out_mask in self.channel_spaces.items():
            subnet_dict[space_id] = self.get_channel_mask_ratio(out_mask,ratio)
        return subnet_dict

    def sample_subnet_ratios(self,ratios):
        """
            ratios:{'space_id':ratio}
        """
        subnet_dict = dict()
        for space_id, out_mask in self.channel_spaces.items():
            subnet_dict[space_id] = self.get_channel_mask_ratio(out_mask,ratios[space_id])
        return subnet_dict

    def sample_subnet(self):
        return self.sample_subnet_ratio_uniform(0.5)

    def set_min_channel(self):
        """Set the number of channels each layer to minimum."""
        pass
    # copied from ratio_pruner.py
    def switch_subnet(self, channel_cfg, subnet_ind=None):
        """Switch the channel config of the supernet according to channel_cfg.

        If we train more than one subnet together, we need to switch the
        channel_cfg from one to another during one training iteration.

        Args:
            channel_cfg (dict): The channel config of a subnet. Key is space_id
                and value is a dict which includes out_channels (and
                in_channels if exists).
            subnet_ind (int, optional): The index of the current subnet. If
                we replace normal BatchNorm2d with ``SwitchableBatchNorm2d``,
                we should switch the index of ``SwitchableBatchNorm2d`` when
                switch subnet. Defaults to None.
        """
        subnet_dict = dict()
        for name, channels_per_layer in channel_cfg.items():
            module = self.name2module[name]
            if (isinstance(module, SwitchableBatchNorm2d)
                    and subnet_ind is not None):
                # When switching bn we should switch index simultaneously
                module.index = subnet_ind
                continue

            out_channels = channels_per_layer['out_channels']
            out_mask = torch.zeros_like(module.out_mask)
            out_mask[:, :out_channels] = 1

            space_id = self.get_space_id(name)
            if space_id in subnet_dict:
                assert torch.equal(subnet_dict[space_id], out_mask)
            elif space_id is not None:
                subnet_dict[space_id] = out_mask

        self.set_subnet(subnet_dict)

    def convert_switchable_bn(self, module, num_bns):
        """Convert normal ``nn.BatchNorm2d`` to ``SwitchableBatchNorm2d``.

        Args:
            module (:obj:`torch.nn.Module`): The module to be converted.
            num_bns (int): The number of ``nn.BatchNorm2d`` in a
                ``SwitchableBatchNorm2d``.

        Return:
            :obj:`torch.nn.Module`: The converted module. Each
                ``nn.BatchNorm2d`` in this module has been converted to a
                ``SwitchableBatchNorm2d``.
        """
        module_output = module
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module_output = SwitchableBatchNorm2d(module.num_features, num_bns)

        for name, child in module.named_children():
            module_output.add_module(
                name, self.convert_switchable_bn(child, num_bns))

        del module
        return module_output
        # copied end