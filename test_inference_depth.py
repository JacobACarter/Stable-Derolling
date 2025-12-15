import argparse
import logging
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from PIL import Image

from omegaconf import OmegaConf
from marigold import MarigoldPipeline
from src.dataset import get_dataset, DatasetMode, BaseRGBDataset
from src.util.config_util import (
    find_value_in_omegaconf,
    recursive_load_config,
)

def _replace_unet_conv_in(model):
    """Replace the first layer to accept 8 input channels"""
    _weight = model.unet.conv_in.weight.clone()
    _bias = model.unet.conv_in.bias.clone()
    _weight = _weight.repeat((1, 2, 1, 1)) * 0.5
    _n_convin_out_channel = model.unet.conv_in.out_channels
    _new_conv_in = Conv2d(8, _n_convin_out_channel, kernel_size=3, stride=1, padding=1)
    _new_conv_in.weight = Parameter(_weight)
    _new_conv_in.bias = Parameter(_bias)
    model.unet.conv_in = _new_conv_in
    model.unet.config["in_channels"] = 8
    logging.info("Replaced UNet conv_in with 8-channel input.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run Marigold inference on training dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to the training YAML config")
    parser.add_argument("--checkpoint", type=str,
                        default="prs-eth/marigold-depth-v1-1",
                        help="Path to UNet checkpoint")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for dataloader")
    parser.add_argument("--GPU", type=int, default=0, help="GPU index")
    parser.add_argument("--half_precision", action="store_true", help="Run in fp16")
    parser.add_argument("--denoise_steps", type=int, default=None)
    parser.add_argument("--ensemble_size", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="./training_inference_output")
    args = parser.parse_args()

    # -------------------- Device --------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # -------------------- Load config --------------------
    cfg = recursive_load_config(args.config)
    train_cfg = cfg.dataset.val[0]
    base_data_dir = os.environ.get("BASE_DATA_DIR", "../rolling-shutter-data")

    # -------------------- Load training dataset --------------------
    train_dataset: BaseRGBDataset = get_dataset(
        train_cfg,
        base_data_dir=base_data_dir,
        mode=DatasetMode.TRAIN
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    logging.info(f"Loaded training dataset with {len(train_dataset)} samples.")

    # -------------------- Load model --------------------
    dtype = torch.float16 if args.half_precision else torch.float32
    pipe = MarigoldPipeline.from_pretrained(args.checkpoint, torch_dtype=dtype)

    if 8 != pipe.unet.config["in_channels"]:
        _replace_unet_conv_in(pipe)

    # pipe.unet.load_state_dict(torch.load(args.checkpoint, map_location=device))
    pipe = pipe.to(device)
    logging.info(f"Loaded UNet checkpoint from {args.checkpoint}")

    # -------------------- Run inference --------------------
    os.makedirs(args.output_dir, exist_ok=True)
    output_dir_npy = os.path.join(args.output_dir, "depth_npy")
    os.makedirs(output_dir_npy, exist_ok=True)

    logging.info(f"Running inference on training dataset...")
    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(train_loader)):
            # Assuming sample contains 'image' tensor
            input_images = sample["rolling_int"].to(device)
            outputs = pipe(
                input_images,
                denoising_steps=args.denoise_steps,
                ensemble_size=args.ensemble_size,
                processing_res = 0,
                match_input_res = True,
                batch_size = 1,
                show_progress_bar = True,
            )
            # Save each output in the batch
            pred = outputs.depth_np  # shape [3, H, W]
            depth_to_save = (pred * 65535.0).astype(np.uint16)
            img = Image.fromarray(depth_to_save, mode="I;16")
            # Convert from CHW -> HWC
            # Use the full relative path for the output filename
            relative_path = sample["global_path"]   # e.g., "folder1/folder2/global.png"
            relative_path = "/".join(relative_path)
            save_path_png = os.path.join(output_dir_npy, relative_path)
            
            # Ensure folder exists
            os.makedirs(os.path.dirname(save_path_png), exist_ok=True)
            img.save(save_path_png)

    logging.info(f"Inference complete. Saved depth predictions to {output_dir_npy}")
