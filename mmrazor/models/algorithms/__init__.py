# Copyright (c) OpenMMLab. All rights reserved.
from .align_method_kd import AlignMethodDistill
from .autoslim import AutoSlim
from .darts import Darts
from .detnas import DetNAS
from .general_distill import GeneralDistill
from .spos import SPOS
from .common_pruning import CommonPruningAlgoritm

__all__ = [
    'AutoSlim', 'AlignMethodDistill', 'Darts', 'SPOS', 'DetNAS',
    'GeneralDistill','CommonPruningAlgoritm'
]
