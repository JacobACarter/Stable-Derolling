# Last modified: 2024-04-30
#
# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# If you use or adapt this code, please attribute to https://github.com/prs-eth/marigold.
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------

import io
import os
import random
import tarfile
from enum import Enum
from typing import Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, Resize

from src.util.depth_transform import DepthNormalizerBase


class DatasetMode(Enum):
    RGB_ONLY = "rgb_only"
    EVAL = "evaluate"
    TRAIN = "train"


class DepthFileNameMode(Enum):
    """Prediction file naming modes"""

    id = 1  # id.png
    rgb_id = 2  # rgb_id.png
    i_d_rgb = 3  # i_d_1_rgb.png
    rgb_i_d = 4


def read_image_from_tar(tar_obj, img_rel_path):
    image = tar_obj.extractfile("./" + img_rel_path)
    image = image.read()
    image = Image.open(io.BytesIO(image))


class BaseRGBDataset(Dataset):
    def __init__(
        self,
        mode: DatasetMode,
        filename_ls_path: str,
        dataset_dir: str,
        disp_name: str,
        name_mode: DepthFileNameMode,
        augmentation_args: dict = None,
        resize_to_hw=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.mode = mode
        # dataset info
        self.filename_ls_path = filename_ls_path
        self.dataset_dir = dataset_dir
        assert os.path.exists(
            self.dataset_dir
        ), f"Dataset does not exist at: {self.dataset_dir}"
        self.disp_name = disp_name
        self.name_mode: DepthFileNameMode = name_mode

        # training arguments
        self.augm_args = augmentation_args
        self.resize_to_hw = resize_to_hw
        print(resize_to_hw)

        # Load filenames
        with open(self.filename_ls_path, "r") as f:
            self.filenames = [
                s.split() for s in f.readlines()
            ]  # [['rgb.png', 'depth.tif'], [], ...]

        # Tar dataset
        self.tar_obj = None
        self.is_tar = (
            True
            if os.path.isfile(dataset_dir) and tarfile.is_tarfile(dataset_dir)
            else False
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        rasters, other = self._get_data_item(index)

        rasters = self._training_preprocess(rasters)
        # merge
        outputs = rasters
        outputs.update(other)

        # Include full relative paths for later use
        outputs["global_path"] = other["global_relative_path"]
        outputs["rolling_path"] = other["rolling_relative_path"]

        return outputs

    def _get_data_item(self, index):
        global_rel_path, rolling_rel_path = self._get_data_path(index=index)

        rasters = {}

        # RGB data (Global Shutter)
        rasters.update(self._load_rgb_data(global_rel_path, False))

        # Rolling data
        rolling_data = self._load_rgb_data(rolling_rel_path, True)
        rasters.update(rolling_data)

        other = {
            "index": index,
            "global_relative_path": global_rel_path,   # e.g., "folder1/folder2/global.png"
            "rolling_relative_path": rolling_rel_path, # e.g., "folder1/folder2/rolling.png"
        }

        return rasters, other




    def _load_rgb_data(self, rgb_rel_path, rolling):
        # Read RGB data
        rgb = self._read_rgb_file(rgb_rel_path)
        rgb_norm = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        if rolling:

            outputs = {
                "rolling_int": torch.from_numpy(rgb).int(),
                "rolling_norm": torch.from_numpy(rgb_norm).float(),
            }
        else:
            # print("Global: " + str(rgb_rel_path))
            outputs = {
                "global_int": torch.from_numpy(rgb).int(),
                "global_norm": torch.from_numpy(rgb_norm).float(),
            }
        return outputs


    def _get_data_path(self, index):
        filename_line = self.filenames[index]

        # Get data path
        global_rel_path = filename_line[0]

        rolling_rel_path = filename_line[1]
        return global_rel_path, rolling_rel_path

    def _read_image(self, img_rel_path) -> np.ndarray:
        if self.is_tar:
            if self.tar_obj is None:
                self.tar_obj = tarfile.open(self.dataset_dir)
            # print(self.tar_obj.members)
            image_to_read = self.tar_obj.extractfile(img_rel_path)
            image_to_read = image_to_read.read()
            image_to_read = io.BytesIO(image_to_read)
        else:
            image_to_read = os.path.join(self.dataset_dir, img_rel_path)
        image = Image.open(image_to_read).convert("RGB")  # [H, W, rgb]
        image = np.asarray(image)
        return image

    def _read_rgb_file(self, rel_path) -> np.ndarray:
        rgb = self._read_image(rel_path)
        ##temp to convert greyscale to rgb
        # w, h = grey.shape
        # rgb = np.empty((w, h, 3), dtype=np.uint8)
        # rgb[:, :, 0] = grey
        # rgb[:, :, 1] = grey
        # rgb[:, :, 2] = grey
        rgb = np.transpose(rgb, (2, 0, 1)).astype(int)  # [rgb, H, W]
        return rgb

    def _training_preprocess(self, rasters):
        # Augmentation
        if self.mode == DatasetMode.TRAIN:
            if self.augm_args is not None:
                rasters = self._augment_data(rasters)

        # Resize
        if self.resize_to_hw is not None:
            # print("here")
            resize_transform = Resize(
                size=self.resize_to_hw, interpolation=InterpolationMode.NEAREST_EXACT
            )
            rasters = {k: resize_transform(v) for k, v in rasters.items()}

        return rasters

    def _augment_data(self, rasters_dict):
        # lr flipping
        lr_flip_p = self.augm_args.lr_flip_p
        if random.random() < lr_flip_p:
            rasters_dict = {k: v.flip(-1) for k, v in rasters_dict.items()}

        return rasters_dict

    def __del__(self):
        if hasattr(self, "tar_obj") and self.tar_obj is not None:
            self.tar_obj.close()
            self.tar_obj = None


def get_pred_name(rgb_basename, name_mode, suffix=".png"):
    if DepthFileNameMode.rgb_id == name_mode:
        pred_basename = "pred_" + rgb_basename.split("_")[1]
    elif DepthFileNameMode.i_d_rgb == name_mode:
        pred_basename = rgb_basename.replace("_rgb.", "_pred.")
    elif DepthFileNameMode.id == name_mode:
        pred_basename = "pred_" + rgb_basename
    elif DepthFileNameMode.rgb_i_d == name_mode:
        pred_basename = "pred_" + "_".join(rgb_basename.split("_")[1:])
    else:
        raise NotImplementedError
    # change suffix
    pred_basename = os.path.splitext(pred_basename)[0] + suffix

    return pred_basename
