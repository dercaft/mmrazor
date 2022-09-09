# Copyright (c) OpenMMLab. All rights reserved.
from .ratio_pruning import RatioPruner
from .structure_pruning import StructurePruner
from .common_pruning import CommonPruner
from .utils import *  # noqa: F401,F403
from .metrics import *

METRICS={
    "L1":L1Norm,
    "FPGM":FPGM,
    "HRANK":Hrank,
    "APOZ":APOZ,
    "CDP":CDP,
}
WEIGHT_METRICS={
    "L1":L1Norm,
    "FPGM":FPGM,
}
FILTER_METRICS={
    "HRANK":Hrank,
    "APOZ":APOZ,
    "CDP":CDP,
}
__all__ = ['RatioPruner', 'StructurePruner', 'CommonPruner']
