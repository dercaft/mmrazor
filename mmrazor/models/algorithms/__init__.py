# Copyright (c) OpenMMLab. All rights reserved.
from .align_method_kd import AlignMethodDistill
from .autoslim import AutoSlim
from .darts import Darts
from .detnas import DetNAS
from .general_distill import GeneralDistill
from .spos import SPOS
from .channel_pruning import ChannelPruningAlgoritm
from .fusion_pruning import FusionPruningAlgoritm
NAS = [
    'Darts', 'SPOS', 'DetNAS'
]
PRUNE = [
    'AutoSlim','ChannelPruningAlgoritm','FusionPruningAlgoritm'
]
MERGE = [
    
]
DISTILL = [
    'GeneralDistill','AlignMethodDistill'
]
ALL_LIST=NAS + PRUNE + MERGE + DISTILL
__all__ = ['Darts', 'SPOS', 'DetNAS', 'AutoSlim', 'ChannelPruningAlgoritm', 'FusionPruningAlgoritm', 'GeneralDistill', 'AlignMethodDistill']
