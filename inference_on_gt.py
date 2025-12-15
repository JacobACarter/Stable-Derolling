import argparse
import logging
import os
import tarfile
import io
from PIL import Image
import numpy as np
import torch
from tqdm.auto import tqdm
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from marigold import MarigoldPipeline

EXTENSION_LIST = [".jpg", ".jpeg", ".png"]

def _replace_unet_conv_in(model):
    _weight = model.unet.conv_in.weight.clone()
    _bias = model.unet.conv_in.bias.clone()
    _weight = _weight.repeat((1, 2, 1, 1)) * 0.5
    _new_conv_in = Conv2d(8, model.unet.conv_in.out_channels, 3, 1, 1)
    _new_conv_in.weight = Parameter(_weight)
    _new_conv_in.bias = Parameter(_bias)
    model.unet.conv_in = _new_conv_in
    model.unet.config["in_channels"] = 8
    logging.info("Unet conv_in layer replaced for 8-channel input.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run Marigold depth estimation from tar archive.")
    parser.add_argument("--checkpoint", type=str, default="prs-eth/marigold-lcm-v1-0")
    parser.add_argument("--input_tar", type=str, required=True, help="Path to tar archive containing images.")
    parser.add_argument("--datasplit", type=str, required=True, help="Path to datasplit file listing relative image paths.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--denoise_steps", type=int, default=None)
    parser.add_argument("--ensemble_size", type=int, default=5)
    parser.add_argument("--half_precision", action="store_true")
    parser.add_argument("--processing_res", type=int, default=None)
    parser.add_argument("--output_processing_res", action="store_true")
    parser.add_argument("--resample_method", choices=["bilinear", "bicubic", "nearest"], default="bilinear")
    parser.add_argument("--color_map", type=str, default="Spectral")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--apple_silicon", action="store_true")
    parser.add_argument("--GPU", type=str, default="0")
    args = parser.parse_args()

    # -------------------- Environment --------------------
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

    if args.apple_silicon:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"device = {device}")

    # -------------------- Output dirs --------------------
    output_dir_color = os.path.join(args.output_dir, "depth_colored")
    output_dir_tif = os.path.join(args.output_dir, "depth_bw")
    output_dir_npy = os.path.join(args.output_dir, "depth_npy")
    os.makedirs(output_dir_color, exist_ok=True)
    os.makedirs(output_dir_tif, exist_ok=True)
    os.makedirs(output_dir_npy, exist_ok=True)

    # -------------------- Load Model --------------------
    dtype = torch.float16 if args.half_precision else torch.float32
    checkpoint_path = args.checkpoint
    pipe: MarigoldPipeline = MarigoldPipeline.from_pretrained(checkpoint_path)
    if pipe.unet.config["in_channels"] != 8:
        _replace_unet_conv_in(pipe)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        pass
    pipe = pipe.to(device)

    # -------------------- Load datasplit --------------------
    with open(args.datasplit, "r") as f:
        rel_paths = [line.strip().split()[1] for line in f if line.strip()]

    logging.info(f"Loaded {len(rel_paths)} image paths from datasplit file.")

    # -------------------- Open tar --------------------
    tar = tarfile.open(args.input_tar, "r:*")

    # -------------------- Inference --------------------
    with torch.no_grad():
        for rel_path in tqdm(rel_paths, desc="Estimating depth"):
            ext = os.path.splitext(rel_path)[1].lower()
            if ext not in EXTENSION_LIST:
                continue

            rel_dir = os.path.dirname(rel_path)
            base_name = os.path.splitext(os.path.basename(rel_path))[0] + "_pred"
            out_color = os.path.join(output_dir_color, rel_dir)
            out_tif = os.path.join(output_dir_tif, rel_dir)
            out_npy = os.path.join(output_dir_npy, rel_dir)
            os.makedirs(out_color, exist_ok=True)
            os.makedirs(out_tif, exist_ok=True)
            os.makedirs(out_npy, exist_ok=True)

            out_color_path = os.path.join(out_color, f"{base_name}_colored.png")
            out_tif_path = os.path.join(out_tif, f"{base_name}.png")
            out_npy_path = os.path.join(out_npy, f"{base_name}.npy")

            # === SKIP IF OUTPUT EXISTS ===
            if all(os.path.exists(p) for p in [out_color_path, out_tif_path, out_npy_path]):
                logging.info(f"Skipping {rel_path}: outputs already exist.")
                continue

            try:
                member = tar.getmember(rel_path)
            except KeyError:
                logging.warning(f"Image {rel_path} not found in tar.")
                continue

            # Load image directly from tar
            with tar.extractfile(member) as f:
                image_bytes = f.read()
                input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # RNG setup
            generator = None if args.seed is None else torch.Generator(device=device).manual_seed(args.seed)

            # Run pipeline
            pipe_out = pipe(
                input_image,
                denoising_steps=args.denoise_steps,
                ensemble_size=args.ensemble_size,
                processing_res=300,
                match_input_res=not args.output_processing_res,
                batch_size=args.batch_size,
                color_map=args.color_map,
                show_progress_bar=False,
                resample_method=args.resample_method,
                generator=generator,
            )

            depth_pred = pipe_out.depth_np
            depth_colored = pipe_out.depth_colored

            # Save npy
            np.save(out_npy_path, depth_pred)

            # Save 16-bit PNG
            depth_16 = (depth_pred * 65535.0).astype(np.uint16)
            Image.fromarray(depth_16).save(out_tif_path, mode="I;16")

            # Save colorized
            depth_colored.save(out_color_path)

    tar.close()
    logging.info("All done.")
