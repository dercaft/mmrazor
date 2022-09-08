# Copyright (c) OpenMMLab. All rights reserved.
from .ratio_pruning import RatioPruner
from .structure_pruning import StructurePruner
from .common_pruning import CommonPruner
from .utils import *  # noqa: F401,F403

__all__ = ['RatioPruner', 'StructurePruner', 'CommonPruner']
