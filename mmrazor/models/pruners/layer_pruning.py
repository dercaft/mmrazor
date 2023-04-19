import copy
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from ordered_set import OrderedSet

from types import MethodType

import torch
import torch.nn as nn

import mmcv
from mmcv import digit_version
# from mmrazor.models.builder import PRUNERS
from ..builder import PRUNERS
from .structure_pruning import StructurePruner

@PRUNERS.register_module()
# class LayerPruner(BaseModule, metaclass=ABCMeta):
class LayerPruner(StructurePruner):
    """
        层剪枝模块
        设计思想: 基于StructurePruner
            但去掉其中mask相关的代码
            用类内变量记录要被删除的层
            剪枝时要将这些层的参数设置为中心为1的矩阵，保证前后的激活图是等价变换
        问题: 如果出现了跨层的连接，那么这个剪枝就不是等价变换了，这个问题如何解决？？？ TODO
        
        self.channel_spaces 记录了所有要被剪枝的层的名字
        StructurePruner里记录模型结构关系的主要变量:
        self.name2module
        self.module2name
        self.module2group
        self.shared_module
        self.node2parents
        self.modules_have_ancest list
        self.modules_have_child OrderedSet
        self.channel_spaces

        测试流程：
        1. 层剪枝，验证剪枝后权重被改变（梯度设置为无），名字被记录，原权重被记录
        2. 恢复，验证剪枝后权重被恢复（梯度设置回来），名字被清空，原权重被清空，模型恢复精度
    """

    def __init__(self, except_start_keys=['head.fc']):
        super(LayerPruner, self).__init__()
        self.record_parameter=dict()
        self.subnet_dict=dict()
    # MFunction
    def prepare_from_supernet(self, supernet):
        """Prepare for pruning."""

        module2name = OrderedDict()
        name2module = OrderedDict()
        var2module = OrderedDict()

        # record the visited module name during trace path
        visited = dict()
        # Record shared modules which will be visited more than once during
        # forward such as shared detection head in RetinaNet.
        # If a module is not a shared module and it has been visited during
        # forward, its parent modules must have been traced already.
        # However, a shared module will be visited more than once during
        # forward, so it is still need to be traced even if it has been
        # visited.
        self.shared_module = []
        tmp_shared_module_hook_handles = list()

        for name, module in supernet.model.named_modules():
            if isinstance(module, nn.GroupNorm):
                min_required_version = '1.6.0' 
                assert digit_version(
                    torch.__version__
                ) >= digit_version(min_required_version), (
                    f'Requires pytorch>={min_required_version} to auto-trace'
                    f'GroupNorm correctly.')
            if hasattr(module, 'weight'):
                # trace shared modules
                module.cnt = 0
                # the handle is only to remove the corresponding hook later
                handle = module.register_forward_hook(
                    self.trace_shared_module_hook)
                tmp_shared_module_hook_handles.append(handle)

                module2name[module] = name
                name2module[name] = module
                var2module[id(module.weight)] = module
                # self.add_pruning_attrs(module)
                visited[name] = False
            # if isinstance(module, SwitchableBatchNorm2d):
            #     name2module[name] = module
        self.name2module = name2module
        self.module2name = module2name

        # Set requires_grad to True. If the `requires_grad` of a module's
        # weight is False, we can not trace this module by parsing backward.
        param_require_grad = dict()
        for param in supernet.model.parameters():
            param_require_grad[id(param)] = param.requires_grad
            param.requires_grad = True

        pseudo_img = torch.randn(1, 3, 224, 224)
        # todo: support two stage detector and mmseg
        pseudo_img = supernet.forward_dummy(pseudo_img)
        pseudo_loss = supernet.cal_pseudo_loss(pseudo_img)

        # `trace_shared_module_hook` and `cnt` are only used to trace the
        # shared modules in a model and need to be remove later
        for name, module in supernet.model.named_modules():
            if hasattr(module, 'weight'):
                del module.cnt

        for handle in tmp_shared_module_hook_handles:
            handle.remove()

        # We set requires_grad to True to trace the whole architecture
        # topology. So it should be reset after that.
        for param in supernet.model.parameters():
            param.requires_grad = param_require_grad[id(param)]
        del param_require_grad

        non_pass_paths = list()
        cur_non_pass_path = list()
        self.trace_non_pass_path(pseudo_loss.grad_fn, module2name, var2module,
                                 cur_non_pass_path, non_pass_paths, visited)

        norm_conv_links = dict()
        self.trace_norm_conv_links(pseudo_loss.grad_fn, module2name,
                                   var2module, norm_conv_links, visited)
        self.norm_conv_links = norm_conv_links

        # a node can be the name of a conv module or a str like 'concat_{id}'
        node2parents = self.find_node_parents(non_pass_paths)
        self.node2parents = node2parents

        same_out_channel_groups = self.make_same_out_channel_groups(
            node2parents, name2module)

        self.module2group = dict()
        for group_name, group in same_out_channel_groups.items():
            for module_name in group:
                self.module2group[module_name] = group_name

        self.modules_have_ancest = list()
        for node_name, parents_name in node2parents.items():
            if node_name in name2module and len(parents_name) > 0:
                self.modules_have_ancest.append(node_name)

        self.modules_have_child = OrderedSet()
        for parents_name in node2parents.values():
            for name in parents_name:
                # The node is a module in supernet
                if name in name2module:
                    self.modules_have_child.add(name)

        self.channel_spaces = self.build_channel_spaces(name2module)

        self._reset_norm_running_stats(supernet)


    # MFunction
    def sample_subnet(self):
        """Sample a subnet from the supernet.
            随机选取一组层来剪枝

        Returns:
            dict: Record the information to build the subnet from the supernet,
                its keys are the properties ``space_id`` in the pruner's search
                spaces, and its values are corresponding sampled out_mask.
        """
        sample=self.name2module.__len__()//3
        for i,key in enumerate(self.name2module):
            if(i>=sample):
                break
            self.subnet_dict[key]=1
        return self.subnet_dict
    # MFunction, Placeholder because of inheriting from StructurePruner
    def prune_layer(self, layer_name:str):
        '''
            Layer Pruner: set the weight of the layer to be a zero matrix with only 1 in center position
        Args:
            layer_name: the name of the layer to be pruned
        Process: 
            1. record the original weight
            2. set the weight to be a zero matrix with only 1 in center position
            3. set the bias to be 0
            4. if have BN, set the mean to be 0, the variance to be 1
        '''
        if self.record_parameter.get(layer_name) is not None:
            print("WARNING: Layer {} has been pruned".format(layer_name))
            return
        if self.name2module.get(layer_name) is None:
            print("WARNING: Layer {} not found".format(layer_name))
            return
        if not hasattr(self.name2module[layer_name],'weight') or self.name2module[layer_name].weight is None:
            print("WARNING: No available weight, MODULE TYPE: {}".format(type(self.name2module[layer_name])))
            return
        
        module=self.name2module[layer_name]
        self.record_parameter[layer_name]=self.record_parameter.setdefault(layer_name,dict())
        if hasattr(module, 'weight') and module.weight is not None:
            self.record_parameter[layer_name]['weight']=module.weight.data.clone()
            module.weight.requires_grad = False
            # if type(module) in [nn.Conv2d,mmcv.cnn.bricks.Conv2d]:
            # Norm Conv (C_out, C_in, W, W) C_out==C_in W==W W%2==1
            if 'conv' in layer_name.lower():
                if module.weight.shape[0]==module.weight.shape[1]:
                    module.weight.data=torch.zero_(module.weight.data)
                    x=module.weight.data.shape[2]//2
                    # _=[module.weight.data[i,i,x,x]=1 for i in range(module.weight.shape[0])]
                    for i in range(module.weight.shape[0]):
                        module.weight.data[i,i,x,x]=1
                # Mobilenet Depth-wise Conv (C_out, 1, W, W) C_in==1 W==W W%2==1
                elif module.weight.shape[1]==1:
                    module.weight.data=torch.zero_(module.weight.data)
                    x=module.weight.data.shape[2]//2
                    for i in range(module.weight.shape[1]):
                        module.weight.data[i,0,x,x]=1
                # Mobilenet Point-wise Conv (1, C_in, 1, 1) C_out==1 W==W==1
                elif module.weight.shape[0]==1:
                    module.weight.data=torch.zero_(module.weight.data)
                    for i in range(module.weight.shape[0]):
                        module.weight.data[0,i,1,1]=1
                else:
                    print("WARNING: Not Supported type, TYPE: {}, SHAPE: {}".format(type(module),module.weight.shape))
            elif 'bn' in layer_name.lower():
                module.weight.data=torch.ones_like(module.weight.data)
                # module.bias.data=torch.zeros_like(module.bias.data)
                module.running_mean.data=torch.zeros_like(module.running_mean.data)
                module.running_var.data=torch.ones_like(module.running_var.data)
        if hasattr(module, 'bias') and module.bias is not None:
            self.record_parameter[layer_name]['bias']=module.bias.data.clone()
            module.bias.requires_grad = False
            module.bias.data=torch.zeros_like(module.bias.data)

        if hasattr(module, 'running_mean'):
            self.record_parameter[layer_name]['running_mean']=module.running_mean.data.clone()
            module.running_mean.data=torch.zeros_like(module.running_mean.data)
        if hasattr(module, 'running_var'):
            self.record_parameter[layer_name]['running_var']=module.running_var.data.clone()
            module.running_var.data=torch.ones_like(module.running_var.data)

    def restore_layer(self,layer_name):
        '''
            Restore the layer to the original state
        '''
        module=self.name2module[layer_name]
        if hasattr(module, 'weight') and module.weight is not None:
            module.weight.requires_grad = True
            module.weight.data=self.record_parameter[layer_name]['weight'].clone()
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.requires_grad = True
            module.bias.data=self.record_parameter[layer_name]['bias'].clone()
        if hasattr(module, 'running_mean'):
            module.running_mean.data=self.record_parameter[layer_name]['running_mean'].clone()
        if hasattr(module, 'running_var'):
            module.running_var.data=self.record_parameter[layer_name]['running_var'].clone()

        del self.record_parameter[layer_name]
    def find_next_bn(self,conv_name):
        '''
            Find the corresponding BN layer of the conv layer
        '''
        conv_name=conv_name.split('.')
        conv_name[-1]='bn'+conv_name[-1][4:]
        bn_name='.'.join(conv_name)
        return bn_name if bn_name in self.name2module else ''

    def set_subnet(self, subnet_dict):
        """Modify the parameter of modules in supernet and record them according to
        subnet_dict.

        Args:
            subnet_dict (dict): the key is name of layers needed to be pruned and the value is Placeholder.

        """
        for module_name in subnet_dict:
            self.prune_layer(module_name)

    # MFunction, Placeholder because of inheriting from StructurePruner
    def export_subnet(self):
        """Generate subnet configs according to the layer pruning record of a
        module."""
        channel_cfg=dict()
        
        return channel_cfg
    
    # MFunction
    def deploy_subnet(self, supernet, channel_cfg):
        """Deploy subnet according `channel_cfg`."""
        pass

    # MFunction
    def build_channel_spaces(self, name2module):
        """Build channel search space.

        Args:
            name2module (dict): A mapping between module_name and module.

        Return:
            dict: The channel search space. The key is space_id and the value
                is the 1 $PlaceHolder$ # corresponding out_mask.
        """
        search_space = dict()

        for module_name in self.modules_have_child:
            need_prune = True
            for key in self.except_start_keys:
                if module_name.startswith(key):
                    need_prune = False
                    break
            if not need_prune:
                continue
            if module_name in self.module2group:
                space_id = self.module2group[module_name]
            else:
                space_id = module_name
            module = name2module[module_name]
            if space_id not in search_space:
                search_space[space_id] = 1

        return search_space

    def set_max_channel(self):
        """Set the number of channels each layer to maximum."""
        print("WARNING: set_max_channel is not implemented for this kind of pruner.")

    def set_min_channel(self):
        """Set the number of channels each layer to minimum."""
        print("WARNING: set_min_channel is not implemented for this kind of pruner.")
        
    @staticmethod
    def modify_forward(module):
        """Modify the forward method of a conv layer."""
        def modified_forward(self, feature):
            return feature
        return MethodType(modified_forward, module)

        # return MethodType(modified_forward, module)

    @staticmethod
    def modify_conv_forward(module):
        """Modify the forward method of a conv layer."""
        def modified_forward(self, feature):
            return feature
        return MethodType(modified_forward, module)

        # return MethodType(modified_forward, module)

    @staticmethod
    def modify_fc_forward(module):
        """Modify the forward method of a linear layer."""
        def modified_forward(self, feature):
            return feature
        return MethodType(modified_forward, module)
        # original_forward = module.forward

        # def modified_forward(self, feature):
        #     if not len(self.in_mask.shape) == len(self.out_mask.shape):
        #         self.in_mask = self.in_mask.reshape(self.in_mask.shape[:2])

        #     feature = feature * self.in_mask
        #     return original_forward(feature)

        # return MethodType(modified_forward, module)

    def add_pruning_attrs(self, module):
        """Add masks to a ``nn.Module``."""
        print("WARNING: add_pruning_attrs is not implemented for this kind of pruner.")
    def skip_module(self,name):
        module=self.name2module[name]
        module.forward=self.modify_forward(module)

    def get_max_channel_bins(self, max_channel_bins):
        """Get the max number of channel bins of all the groups which can be
        pruned during searching.

        Args:
            max_channel_bins (int): The max number of bins in each layer.
        """
        print("WARNING: get_max_channel_bins is not implemented for this kind of pruner.")
        channel_bins_dict = dict()
        # for space_id in self.channel_spaces.keys():
            # channel_bins_dict[space_id] = torch.ones((max_channel_bins, ))
        return channel_bins_dict

    def set_channel_bins(self, channel_bins_dict, max_channel_bins):
        """Set subnet according to the number of channel bins in a layer.
            
        Args:
            channel_bins_dict (dict): The number of bins in each layer. Key is
                the space_id of each layer and value is the corresponding
                mask of channel bin.
            max_channel_bins (int): The max number of bins in each layer.
        """
        print("WARNING: set_channel_bins is not implemented for this kind of pruner.")

        # subnet_dict = dict()
        # for space_id, bin_mask in channel_bins_dict.items():
        #     mask = self.channel_spaces[space_id]
        #     shape = mask.shape
        #     channel_num = shape[1]
        #     channels_per_bin = channel_num // max_channel_bins
        #     new_mask = []
        #     for mask in bin_mask:
        #         new_mask.extend([1] * channels_per_bin if mask else [0] *
        #                         channels_per_bin)
        #     new_mask.extend([0] * (channel_num % max_channel_bins))
        #     new_mask = torch.tensor(new_mask).reshape(*shape)
        #     subnet_dict[space_id] = new_mask
        # self.set_subnet(subnet_dict)