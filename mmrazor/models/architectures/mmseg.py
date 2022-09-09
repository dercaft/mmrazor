# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import ARCHITECTURES
from .base import BaseArchitecture


@ARCHITECTURES.register_module()
class MMSegArchitecture(BaseArchitecture):
    """Architecture based on MMSeg."""

    def __init__(self, **kwargs):
        super(MMSegArchitecture, self).__init__(**kwargs)
    def cal_pseudo_loss(self, pseudo_img):
        """ Used for executing ``forward`` with pseudo_img """
        out=pseudo_img.sum()
        return out