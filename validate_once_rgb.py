# Validate-only script for Marigold model
# Last modified: 2025-10-13
#
# Loads datasets exactly like the main training script, loads checkpoint, and
# runs validation once (saves PNGs for each image).
#
# Usage:
#   python validate_once.py --config config/train_marigold.yaml \
#       --ckpt output/train_X4K_center/checkpoint/latest \
#       --output_dir output/val_results --GPU 0

import argparse
import os
import logging
from datetime import datetime
from typing import List

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from marigold.derolling_pipeline import MarigoldRGBPipeline
from src.dataset import get_dataset, DatasetMode
from src.trainer import get_trainer_cls
from src.util.config_util import recursive_load_config
from src.util.logging_util import config_logging
from src.util.slurm_util import is_on_slurm, get_local_scratch_dir
from tqdm import tqdm
import shutil


def run_validation_once(cfg_path, ckpt_path, output_dir, gpu_num=0, no_cuda=False):
    t_start = datetime.now()
    print(f"start validation at {t_start}")

    # -------------------- Config --------------------
    cfg = recursive_load_config(cfg_path)
    cfg_data = cfg.dataset

    # -------------------- Device --------------------
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
    cuda_avail = torch.cuda.is_available() and not no_cuda
    device = torch.device("cuda" if cuda_avail else "cpu")
    logging.info(f"device = {device}")

    # -------------------- Directories --------------------
    os.makedirs(output_dir, exist_ok=True)
    out_dir_eval = os.path.join(output_dir, "evaluation")
    out_dir_vis = os.path.join(output_dir, "visualization")
    os.makedirs(out_dir_eval, exist_ok=True)
    os.makedirs(out_dir_vis, exist_ok=True)

    # -------------------- Logging --------------------
    config_logging(cfg.logging, out_dir=output_dir)

    # -------------------- Base directories --------------------
    base_data_dir = os.environ.get("BASE_DATA_DIR", "../rolling-shutter-data")
    base_ckpt_dir = os.environ.get("BASE_CKPT_DIR", "../")

    # -------------------- Copy data to local scratch (if Slurm) --------------------
    if is_on_slurm():
        original_data_dir = base_data_dir
        base_data_dir = os.path.join(get_local_scratch_dir(), "Marigold_data")

        required_data_list = []
        from src.util.config_util import find_value_in_omegaconf
        required_data_list = find_value_in_omegaconf("dir", cfg_data)
        required_data_list = list(set(required_data_list))

        logging.info(f"Copying data to local scratch: {required_data_list}")
        for d in tqdm(required_data_list, desc="Copy data to scratch"):
            ori_dir = os.path.join(original_data_dir, d)
            dst_dir = os.path.join(base_data_dir, d)
            os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
            if os.path.isfile(ori_dir):
                shutil.copyfile(ori_dir, dst_dir)
            elif os.path.isdir(ori_dir):
                shutil.copytree(ori_dir, dst_dir)
        logging.info(f"Data copied to {base_data_dir}")

    # -------------------- Transforms --------------------
    transforms = v2.Compose([
        v2.Resize(size=(300, 300)),
        v2.RandomHorizontalFlip(p=0.5),
    ])

    # -------------------- Validation datasets --------------------
    val_loaders: List[DataLoader] = []
    for _val_dic in cfg_data.val:
        _val_dataset = get_dataset(
            _val_dic,
            base_data_dir=base_data_dir,
            mode=DatasetMode.EVAL,
            transforms=transforms,
        )
        _val_loader = DataLoader(
            dataset=_val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.dataloader.num_workers,
        )
        val_loaders.append(_val_loader)
    print(f"Validation datasets: {[len(v.dataset) for v in val_loaders]}")

    # -------------------- Model --------------------
    model = MarigoldRGBPipeline.from_pretrained(
        os.path.join(base_ckpt_dir, cfg.model.pretrained_path)
        if "pretrained_path" in cfg.model
        else "../stable-diffusion-2"
    )
    model.unet.to(device)

    # -------------------- Trainer --------------------
    trainer_cls = get_trainer_cls(cfg.trainer.name)
    trainer = trainer_cls(
        cfg=cfg,
        model=model,
        train_dataloader=None,
        device=device,
        base_ckpt_dir=base_ckpt_dir,
        out_dir_ckpt=os.path.join(output_dir, "checkpoint"),
        out_dir_eval=out_dir_eval,
        out_dir_vis=out_dir_vis,
        accumulation_steps=1,
        val_dataloaders=val_loaders,
        vis_dataloaders=val_loaders,
    )

    # -------------------- Load Checkpoint --------------------
    trainer.load_checkpoint(ckpt_path, load_trainer_state=False)
    print(f"Loaded checkpoint from {ckpt_path}")

    # -------------------- Run Validation Once --------------------
    for val_loader in val_loaders:
        dataset_name = getattr(val_loader.dataset, "disp_name", "val_dataset")
        vis_dir = os.path.join(out_dir_vis, dataset_name)
        os.makedirs(vis_dir, exist_ok=True)
        print(f"Running validation for {dataset_name}...")
        trainer.validate_single_dataset(val_loader, save_to_dir=vis_dir)
        print(f"✅ Saved outputs to {vis_dir}")

    print("✅ Validation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run validation only and save PNGs")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint directory (containing trainer.ckpt)")
    parser.add_argument("--output_dir", required=True, help="Directory to save validation results")
    parser.add_argument("--GPU", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true", help="Do not use CUDA")
    args = parser.parse_args()

    run_validation_once(
        cfg_path=args.config,
        ckpt_path=args.ckpt,
        output_dir=args.output_dir,
        gpu_num=args.GPU,
        no_cuda=args.no_cuda,
    )
