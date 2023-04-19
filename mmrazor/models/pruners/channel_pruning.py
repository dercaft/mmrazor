# Copyright (c) OpenMMLab. All rights reserved.
from pydoc import doc
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules import GroupNorm

from mmrazor.models.builder import PRUNERS
from .structure_pruning import StructurePruner
from .utils import SwitchableBatchNorm2d
from .metrics import get_metric, WEIGHT_METRICS, FILTER_METRICS

@PRUNERS.register_module()
class ChannelPruner(StructurePruner):
    """A pruner copied from ratio_pruner.py - RationPruner()

    """

    def __init__(self, **kwargs):
        self.pruning_metric_name=kwargs.get("pruning_metric_name","L2")
        if kwargs.get("pruning_metric_name",None) is not None:
            kwargs.__delitem__("pruning_metric_name")
        super(ChannelPruner, self).__init__(**kwargs)
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
        super(ChannelPruner, self).prepare_from_supernet(supernet)
        self.space2name={}
        for name,module in supernet.model.named_modules():
            name=name.lstrip('model.')
            spi=self.get_space_id(name)
            if not isinstance(spi,str): continue
            g=self.space2name.setdefault(spi,[])
            g.append(name)
            self.space2name[spi]=g
        self.name2space={}
        for k,v in self.space2name.items():
            for name in v:
                self.name2space[name]=k
        
    def get_channel_mask_ratio_randomly(self,out_mask):
        """ Randomly choose a width ratio of a layer
            out_mask: (1,channels,1,1)
        """ 
        while True:
            random_ratio=1-np.random.rand()
            if random_ratio>1/out_mask.size(1): break
        new_out_mask=self.get_channel_mask_ratio_metric(out_mask=out_mask,ratio=random_ratio)
        return new_out_mask
    def get_channel_mask_ratio_metric(self,out_mask,ratio,metric=None,**kwargs):
        """ Denote ratio and metric to create a new mask.
            out_mask: torch.Tensor shape(1,out_channels,1,1)
            ratio: float(0,1]
            metric: torch.Tensor shape(out_channels,) # Bigger, better
        """
        assert isinstance(ratio,int) or ratio>0 and ratio<=1, f'Ratio out of range, now is: {ratio}'
        if metric:
            assert isinstance(metric,torch.Tensor) and len(metric) == out_mask.size(1), \
                f'Metric length is mismatch with out_mask,' \
                f'Metric size is: {metric.size()}, out_mask size is: {out_mask.size()}'
        # new_channels=int(round(out_mask.size(1)*ratio))
        # assert new_channels >0, f'Output channels should be a positive integer. ratio is {ratio}.'
        if isinstance(ratio,int):
            new_channels=ratio
        else:
            new_channels=int(np.ceil(out_mask.size(1)*ratio))

        new_out_mask=torch.zeros_like(out_mask)
        if metric:
            new_indx=torch.topk(metric,new_channels)[1]
            new_out_mask.index_fill_(dim=1,index=new_indx,value=1)
        elif kwargs.get('random',False):
            # randomly generate a mask with specified remaining index
            new_indx=torch.randperm(out_mask.size(1))[:new_channels]
            new_out_mask.index_fill_(dim=1,index=new_indx,value=1)
        else:
            new_out_mask[:,:new_channels]=1
        return new_out_mask
    # def get_channel_mask_discrete_metric(self,out_mask,ratio,metric=None):
    #     '''
    #         ratio is int
    #     '''
    #     assert isinstance(ratio,int) and ratio>0, f'Ratio should be a positive integer. ratio is {ratio}.'
    #     new_channels=ratio
    #     new_out_mask=torch.zeros_like(out_mask)
    #     if metric:
    #     pass
    def sample_subnet_ratio_randomly(self):
        """Random sample subnet by random mask.

        Returns:
            dict: Record the information to build the subnet from the supernet,
                its keys are the properties ``space_id`` in the pruner's search
                spaces, and its values are corresponding sampled out_mask.
        """
        subnet_dict = dict()
        for space_id, out_mask in self.channel_spaces.items():
            subnet_dict[space_id] = self.get_channel_mask_ratio_randomly(out_mask)
        return subnet_dict

    def sample_subnet_uniform(self,ratio,metric=None):
        subnet_dict = dict()
        for space_id, out_mask in self.channel_spaces.items():
            subnet_dict[space_id] = self.get_channel_mask_ratio_metric(out_mask,ratio,metric=metric)
        return subnet_dict

    def sample_subnet_nonuni(self,ratios,metrics=None):
        """
            ratios:{'space_id':ratio}
            metrics:{'space_id':metric}
        """
        subnet_dict = dict()
        for space_id, out_mask in self.channel_spaces.items():
            metric = None if isinstance(metrics,type(None)) else metrics[space_id]
            subnet_dict[space_id] = self.get_channel_mask_ratio_metric(out_mask,ratios[space_id],metric=metric)
        return subnet_dict
    def sample_subnet_diff(self,ratio,metrics):
        """
            ratio:float
            metrics:{'space_id':metric[]}
        """
        subnet_dict = dict()
        # unzip metrics into a list [('spece_id',metric)]
        global_pairs=[(key,i,s) for key,met in metrics.items() for i,s in enumerate(met)]
        global_pairs=sorted(global_pairs,key=lambda x:x[2],reverse=True)
        global_pairs=global_pairs[:int(len(global_pairs)*ratio)]
        # form index dict based on global_pairs
        index_dict={}
        for key,i,s in global_pairs:
            index_dict.setdefault(key,[]).append(i)
        # form subnet_dict
        for space_id, out_mask in self.channel_spaces.items():
            new_out_mask=torch.zeros_like(out_mask)
            try:
                new_out_mask.index_fill_(dim=1,index=torch.tensor(index_dict[space_id]),value=1)
            except:
                print(f'WARNING: No metric for {space_id}, set to 0.')
            subnet_dict[space_id] = new_out_mask
        return subnet_dict
    def sample_subnet(self,type:str="uniform",ratio=0.5):
        """Sample a subnet from the supernet, globally uniform or randomly.
        """
        if type=="uniform":
            return self.sample_subnet_uniform(ratio)
        elif type=="random":
            return self.sample_subnet_ratio_randomly()
        else:
            raise NotImplementedError

    def set_min_channel(self):
        """Set the number of channels each layer to minimum."""
        subnet_dict=self.sample_subnet()
        self.set_subnet(subnet_dict)

    def get_metric_local(self,metric_name,contents):
        """Get metric of each layer.

        Args:
            metric_name (str): The name of the metric.
            contents (dict): The contents of measured contents, including weights and features.
                {'layer_name'(str):'weight|feature'(tensor)}
        Returns:
            dict: The metric of each layer.
        """
        metric_dict = dict()
        layer2score = dict()
        measurement=get_metric(metric_name)
        for space_id, out_mask in self.channel_spaces.items():
            # collect metrics from all layers in the same space, and average them
            met=[]
            for name in self.space2name[space_id]:
                score=measurement(contents[name])
                met.append(score)
                layer2score[name]=score
            metric_dict[space_id]=torch.mean(torch.stack(met,dim=0),dim=0)
        return metric_dict,layer2score
    def remove_denoted_group(self,names:list=[]):
        '''
            Remove named layer from channel_spaces, name2space, space2name
        '''
        remove_list=[]
        remove_subdict={}
        for space in self.channel_spaces.keys():
            # use self.space2name[space] to find if there is any same name
            for name in names:
                a=sum([1 if name in n else 0 for n in self.space2name[space]])
                if a:
                    remove_list.append(space)
                    break
        for space in remove_list:
            remove_subdict[space]=self.channel_spaces.pop(space)
            names=self.space2name.pop(space)
            for n in names:
                self.name2space.pop(n)
        # use remove_list as dict's keys
        # create deleted keys to out_mask

        return remove_subdict

        # for name in names:
        #     self.channel_spaces
    # # copied from ratio_pruner.py
    # def switch_subnet(self, channel_cfg, subnet_ind=None):
    #     """Switch the channel config of the supernet according to channel_cfg.

    #     If we train more than one subnet together, we need to switch the
    #     channel_cfg from one to another during one training iteration.

    #     Args:
    #         channel_cfg (dict): The channel config of a subnet. Key is space_id
    #             and value is a dict which includes out_channels (and
    #             in_channels if exists).
    #         subnet_ind (int, optional): The index of the current subnet. If
    #             we replace normal BatchNorm2d with ``SwitchableBatchNorm2d``,
    #             we should switch the index of ``SwitchableBatchNorm2d`` when
    #             switch subnet. Defaults to None.
    #     """
    #     subnet_dict = dict()
    #     for name, channels_per_layer in channel_cfg.items():
    #         module = self.name2module[name]
    #         if (isinstance(module, SwitchableBatchNorm2d)
    #                 and subnet_ind is not None):
    #             # When switching bn we should switch index simultaneously
    #             module.index = subnet_ind
    #             continue

    #         out_channels = channels_per_layer['out_channels']
    #         out_mask = torch.zeros_like(module.out_mask)
    #         out_mask[:, :out_channels] = 1

    #         space_id = self.get_space_id(name)
    #         if space_id in subnet_dict:
    #             assert torch.equal(subnet_dict[space_id], out_mask)
    #         elif space_id is not None:
    #             subnet_dict[space_id] = out_mask

    #     self.set_subnet(subnet_dict)

    # def convert_switchable_bn(self, module, num_bns):
    #     """Convert normal ``nn.BatchNorm2d`` to ``SwitchableBatchNorm2d``.

    #     Args:
    #         module (:obj:`torch.nn.Module`): The module to be converted.
    #         num_bns (int): The number of ``nn.BatchNorm2d`` in a
    #             ``SwitchableBatchNorm2d``.

    #     Return:
    #         :obj:`torch.nn.Module`: The converted module. Each
    #             ``nn.BatchNorm2d`` in this module has been converted to a
    #             ``SwitchableBatchNorm2d``.
    #     """
    #     module_output = module
    #     if isinstance(module, nn.modules.batchnorm._BatchNorm):
    #         module_output = SwitchableBatchNorm2d(module.num_features, num_bns)

    #     for name, child in module.named_children():
    #         module_output.add_module(
    #             name, self.convert_switchable_bn(child, num_bns))

    #     del module
    #     return module_output
    #     # copied end