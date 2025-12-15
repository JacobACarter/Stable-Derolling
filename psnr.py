from glob import glob
import os
import numpy as np
import cv2
from math import log10
from skimage.metrics import structural_similarity as compare_ssim
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


# ---------------- Perceptual Loss ----------------
class PerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
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
        feat1 = self.vgg(img1)
        feat2 = self.vgg(img2)
        return torch.nn.functional.mse_loss(feat1, feat2)


# ---------------- Helper functions ----------------
def is_image_file(path):
    return os.path.isfile(path) and path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))

def mse_numpy(pred, target):
    return np.mean((pred - target) ** 2)

def PSNR(original, compressed):
    original = original.astype(np.float32)
    compressed = compressed.astype(np.float32)
    mse = mse_numpy(original, compressed)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 1.0
    psnr = 10 * np.log10((PIXEL_MAX ** 2) / mse)
    return psnr


# ---------------- Depth metrics ----------------
def abs_rel(pred, gt, mask=None):
    if mask is None:
        mask = (gt > 0)
    pred = pred[mask]
    gt = gt[mask]
    return np.mean(np.abs(pred - gt) / np.clip(gt, 1e-6, None))

def d1_metric(pred, gt, mask=None, thresh=1.25):
    if mask is None:
        mask = (gt > 0)
    pred = pred[mask]
    gt = gt[mask]
    ratio = np.maximum(pred / np.clip(gt, 1e-6, None), gt / np.clip(pred, 1e-6, None))
    bad = ratio > thresh
    return np.mean(bad)


# ---------------- Paths ----------------
#"metric_depth/fine-tuned/depths"
output_depth = "marigold_alone/depth_npy/10-12/subset_cropped"
gt_depth = "output/10-12-depth/validation/depth_bw/10-12/subset_cropped"

gt_list = sorted([f for f in glob(os.path.join(gt_depth, "**", "*"), recursive=True) if is_image_file(f)])
out_list = sorted([f for f in glob(os.path.join(output_depth, "**", "*"), recursive=True) if is_image_file(f)])

print(f"Found {len(gt_list)} GT and {len(out_list)} outputs")

# ---------------- Metrics Init ----------------
out_psnr = 0
ssim_output_cumulative = 0
mse_output_cumulative = 0
loss_val_cumulative = 0
absrel_cumulative = 0
d1_cumulative = 0
inc = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = PerceptualLoss().to(device)
print(f"Using device: {device}")

# ---------------- Evaluation Loop ----------------
for i, out_path in enumerate(out_list):
    try:
        output_rel = os.path.relpath(out_path, output_depth)
        # Extract top-level data folder (first folder after output_depth)
        data_folder_out = output_rel.split(os.sep)[0]
        print(output_rel)

        # Only consider GT files in the same data folder
        gt_candidates = [p for p in gt_list if os.path.relpath(p, gt_depth).split(os.sep)[0] == data_folder_out]
        print(gt_candidates[0])
        print(os.path.basename(gt_candidates[0]))
        # Then match by filename within that data folder
        name = os.path.basename(out_path).replace(".png", "_pred.png")
        print(name)
        print()
        gt_matches = [p for p in gt_candidates if os.path.basename(p) == name]

        if not gt_matches:
            continue

        gt_path = gt_matches[0]

        print(out_path, gt_path)
        output_img = cv2.imread(out_path, cv2.IMREAD_UNCHANGED)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)

        if output_img is None or gt_img is None:
            continue

        output_img = output_img.astype(np.float32) / 255.0
        gt_img = gt_img.astype(np.float32) / 255.0

        # If multi-channel, convert to single depth
        if output_img.ndim == 3:
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
        if gt_img.ndim == 3:
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)

    except Exception as e:
        print(f"Error on {out_path}: {e}")
        continue

    inc += 1
    # MSE, PSNR, SSIM
    mse_output_cumulative += mse_numpy(gt_img, output_img)
    (scoreo, _) = compare_ssim(gt_img, output_img, full=True, data_range=1)
    ssim_output_cumulative += scoreo
    psnr2 = PSNR(gt_img, output_img)
    out_psnr += psnr2

    # AbsRel, D1
    absrel_val = abs_rel(output_img, gt_img)
    d1_val = d1_metric(output_img, gt_img)
    absrel_cumulative += absrel_val
    d1_cumulative += d1_val

    # Perceptual Loss
    gt_tensor = torch.from_numpy(gt_img).unsqueeze(0).unsqueeze(0).to(device)
    out_tensor = torch.from_numpy(output_img).unsqueeze(0).unsqueeze(0).to(device)
    loss_val = loss_fn(gt_tensor, out_tensor)
    loss_val_cumulative += loss_val.item()

    print(f"[{i+1}/{len(out_list)}] {name} | PSNR {psnr2:.3f} | SSIM {scoreo:.3f} | absRel {absrel_val:.3f} | D1 {d1_val:.3f}")

# ---------------- Final Metrics ----------------
if inc > 0:
    print("\n=== Averages over dataset ===")
    print(f"PSNR: {out_psnr / inc:.4f}")
    print(f"SSIM: {ssim_output_cumulative / inc:.4f}")
    print(f"MSE: {mse_output_cumulative / inc:.6f}")
    print(f"absRel: {absrel_cumulative / inc:.4f}")
    print(f"D1: {d1_cumulative / inc:.4f}")
    print(f"Perceptual Loss: {loss_val_cumulative / inc:.6f}")
else:
    print("No valid images were processed.")
