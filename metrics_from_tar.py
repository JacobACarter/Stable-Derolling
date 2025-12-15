#!/usr/bin/env python3
"""
Evaluate metrics for image pairs inside a tar file and compute optical-flow-based
statistics between input (camera2) and GT (camera1).

Now also records maximum optical flow magnitude (% of image diagonal).
"""

import tarfile
import io
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as compare_ssim
import math
import csv
import argparse
import os
from tqdm import tqdm

# ---------------- Perceptual Loss ----------------
class PerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features[:16].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.resize = resize
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)) if resize else transforms.Lambda(lambda x: x),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def forward(self, img1, img2):
        if img1.shape[1] == 1:
            img1 = img1.repeat(1, 3, 1, 1)
            img2 = img2.repeat(1, 3, 1, 1)
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        f1, f2 = self.vgg(img1), self.vgg(img2)
        return torch.nn.functional.mse_loss(f1, f2)

# ---------------- Metrics ----------------
def mse_numpy(pred, target):
    return np.mean((pred - target) ** 2)

def PSNR_dynamic_range(original, compressed):
    original = original.astype(np.float32)
    compressed = compressed.astype(np.float32)
    combined = np.concatenate([original.flatten(), compressed.flatten()])
    min_val, max_val = np.min(combined), np.max(combined)
    if max_val - min_val < 1e-8:
        return np.nan
    original_norm = (original - min_val) / (max_val - min_val)
    compressed_norm = (compressed - min_val) / (max_val - min_val)
    mse = np.mean((original_norm - compressed_norm) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)

def abs_rel(pred, gt):
    mask = (gt > 0)
    pred_masked, gt_masked = pred[mask], gt[mask]
    if pred_masked.size == 0:
        return np.nan
    return np.mean(np.abs(pred_masked - gt_masked) / np.clip(gt_masked, 1e-6, None))

def d1_metric(pred, gt, thresh=1.25):
    mask = (gt > 0)
    pred_masked, gt_masked = pred[mask], gt[mask]
    if pred_masked.size == 0:
        return np.nan
    ratio = np.maximum(pred_masked / np.clip(gt_masked, 1e-6, None),
                       gt_masked / np.clip(pred_masked, 1e-6, None))
    return np.mean(ratio > thresh)

# ---------------- Tar Reading ----------------
def read_image_from_tar(tar, filepath, grayscale=True):
    try:
        member = tar.getmember(filepath)
        f = tar.extractfile(member)
        data = f.read()
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = img.astype(np.float32) / 255.0 if grayscale else img
        return img
    except KeyError:
        return None

# ---------------- Optical Flow ----------------
def compute_flow(gt_img, inp_img):
    """
    Compute optical flow (dx, dy) from input -> gt
    Returns:
        flow_chw: np.array (2,H,W)
        mean_pct, median_pct, pct_above_list, max_pct
    """
    if gt_img.ndim == 3:
        gt_gray = cv2.cvtColor((gt_img*255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        gt_gray = (gt_img*255).astype(np.uint8) if gt_img.dtype != np.uint8 else gt_img

    if inp_img.ndim == 3:
        inp_gray = cv2.cvtColor((inp_img*255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        inp_gray = (inp_img*255).astype(np.uint8) if inp_img.dtype != np.uint8 else inp_img

    flow = cv2.calcOpticalFlowFarneback(
        inp_gray, gt_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    flow_chw = flow.transpose(2, 0, 1)

    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    h, w = mag.shape
    diag = math.sqrt(h*h + w*w)
    mag_pct = (mag / diag) * 100.0

    mean_pct = float(np.mean(mag_pct))
    median_pct = float(np.median(mag_pct))

    # Compute cumulative percentages
    thresholds = [2, 4, 6, 8, 10]  # can adjust or add more
    pct_above_list = [(mag_pct > t).mean() * 100.0 for t in thresholds]

    max_pct = float(np.max(mag_pct))
    return flow_chw, mean_pct, median_pct, pct_above_list, max_pct



# ---------------- Main Evaluation ----------------
def evaluate_from_tar(tar_path, filelist_path, csv_out=None, flow_out_root=None, debug=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = PerceptualLoss().to(device)
    print(f"Using device: {device}")

    psnr_list, ssim_list, mse_list, absrel_list, d1_list, perceptual_list = [], [], [], [], [], []
    flow_mean_pct_list, flow_median_pct_list, flow_above_list, flow_max_pct_list = [], [], [], []

    with tarfile.open(tar_path, "r") as tar:
        with open(filelist_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        for i, line in enumerate(tqdm(lines, desc="Evaluating")):
            parts = line.split()
            if len(parts) != 2:
                if debug:
                    print(f"Skipping malformed line: {line}")
                continue
            gt_path, comp_path = parts
            input_path = gt_path.replace("camera1", "camera2")

            gt_img = read_image_from_tar(tar, gt_path, grayscale=True)
            inp_img_color = read_image_from_tar(tar, input_path, grayscale=False)
            comp_img = read_image_from_tar(tar, comp_path, grayscale=True)

            if gt_img is None or inp_img_color is None or comp_img is None:
                if debug:
                    print(f"Missing files, skipping: {gt_path}")
                continue

            h, w = gt_img.shape
            if comp_img.shape != (h, w):
                comp_img = cv2.resize((comp_img*255).astype(np.uint8), (w, h)).astype(np.float32)/255.0

            mse = mse_numpy(comp_img, gt_img)
            ssim_val, _ = compare_ssim(gt_img, comp_img, full=True, data_range=1.0)
            psnr_val = PSNR_dynamic_range(gt_img, comp_img)
            absrel = abs_rel(comp_img, gt_img)
            d1 = d1_metric(comp_img, gt_img)
            gt_t = torch.from_numpy(gt_img).unsqueeze(0).unsqueeze(0).to(device)
            out_t = torch.from_numpy(comp_img).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                perceptual = loss_fn(gt_t, out_t).item()

            flow_chw, mean_pct, median_pct, flow_hist_pct, max_pct = compute_flow(gt_img, inp_img_color)
            if flow_out_root is not None:
                flow_out_path = os.path.join(flow_out_root, input_path + ".npy")
                os.makedirs(os.path.dirname(flow_out_path), exist_ok=True)
                np.save(flow_out_path, flow_chw)

            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
            mse_list.append(mse)
            absrel_list.append(absrel)
            d1_list.append(d1)
            perceptual_list.append(perceptual)
            flow_mean_pct_list.append(mean_pct)
            flow_median_pct_list.append(median_pct)
            flow_above_list.append(flow_hist_pct)
            flow_max_pct_list.append(max_pct)

            if debug:
                print(f"[{i+1}/{len(lines)}] {comp_path} | PSNR {psnr_val:.3f} | "
                      f"SSIM {ssim_val:.3f} | flow mean% {mean_pct:.3f} | max% {max_pct:.3f}")

    n = len(psnr_list)
    print("\n=== Dataset averages ===")
    if n == 0:
        print("No valid images processed.")
        return
    print(f"Count: {n}")
    print(f"PSNR: {np.nanmean(psnr_list):.4f}")
    print(f"SSIM: {np.nanmean(ssim_list):.4f}")
    print(f"MSE: {np.nanmean(mse_list):.6f}")
    print(f"absRel: {np.nanmean(absrel_list):.4f}")
    print(f"D1: {np.nanmean(d1_list):.4f}")
    print(f"Perceptual Loss: {np.nanmean(perceptual_list):.6f}")
    print(f"Flow mean % of diagonal: {np.nanmean(flow_mean_pct_list):.6f}")
    print(f"Flow median % of diagonal: {np.nanmean(flow_median_pct_list):.6f}")
    flow_pct_above_array = np.array(flow_above_list)  # shape: (num_images, num_thresholds)
    avg_pct_above = np.nanmean(flow_pct_above_array, axis=0)
    thresholds = [2, 4, 6, 8, 10]
    # Print average percentages
    for t, pct in zip(thresholds, avg_pct_above):
        print(f"Average flow >{t}% pixels: {pct:.2f}%")
    print(f"Flow max % of diagonal: {np.nanmean(flow_max_pct_list):.6f}")

    if csv_out:
        with open(csv_out, "w", newline="") as csvf:
            writer = csv.writer(csvf)
            writer.writerow([
                "gt_path","comp_path","PSNR","SSIM","MSE","absRel","D1","Perceptual",
                "flow_mean_pct","flow_median_pct","flow_pct_above_4","flow_max_pct"
            ])
            for i, line in enumerate(lines):
                parts = line.split()
                if len(parts) != 2 or i >= n:
                    continue
                writer.writerow([
                    parts[0], parts[1],
                    psnr_list[i], ssim_list[i], mse_list[i],
                    absrel_list[i], d1_list[i], perceptual_list[i],
                    flow_mean_pct_list[i], flow_median_pct_list[i],
                    flow_above4_list[i], flow_max_pct_list[i]
                ])
        print(f"Wrote per-image metrics to {csv_out}")

# ---------------- Entry Point ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate metrics and optical flow in tar.")
    parser.add_argument("--tar", required=True, help="Path to tar file with images")
    parser.add_argument("--filelist", required=True, help="File listing GT and comp images")
    parser.add_argument("--csv", required=False, help="Optional CSV output")
    parser.add_argument("--debug", action="store_true", help="Enable debug prints")
    parser.add_argument("--flow_out", required=False, default="flows", help="Directory to save flow .npy files")
    args = parser.parse_args()

    evaluate_from_tar(args.tar, args.filelist, csv_out=args.csv, flow_out_root=args.flow_out, debug=args.debug)
