# Author: Bingxin Ke
# Last modified: 2024-05-17

from .marigold_trainer import MarigoldTrainer
from .marigold_rgb_trainer import MarigoldRGBTrainer
from .marigold_rgb_depth_trainer import MarigoldRGBDepthTrainer
from .marigold_dual_unet_trainer import MarigoldDualUnetTrainer
from .marigold_alone_depth_trainer import MarigoldAloneDepthTrainer


trainer_cls_name_dict = {
    "MarigoldTrainer": MarigoldTrainer,
    "MarigoldRGBTrainer": MarigoldRGBTrainer,
    "MarigoldRGBDepthTrainer":MarigoldRGBDepthTrainer,
    "MarigoldDualUnetTrainer":MarigoldDualUnetTrainer,
    "MarigoldAloneDepthTrainer":MarigoldAloneDepthTrainer,
}


def get_trainer_cls(trainer_name):
    return trainer_cls_name_dict[trainer_name]
