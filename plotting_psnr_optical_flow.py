# import os
# import cv2
# import tarfile
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from glob import glob
# from math import log10
# from tqdm import tqdm


# # ---------- Helper Functions ----------

# def is_image_file(path):
#     return path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))

# def PSNR(img1, img2):
#     img1 = img1.astype(np.float32) / 255.0
#     img2 = img2.astype(np.float32) / 255.0
#     mse = np.mean((img1 - img2) ** 2)
#     if mse == 0:
#         return float('inf')
#     return 10 * log10(1.0 / mse)

# def read_tar_image(tar, member_name):
#     try:
#         f = tar.extractfile(member_name)
#         if f is None:
#             return None
#         data = np.frombuffer(f.read(), np.uint8)
#         img = cv2.imdecode(data, cv2.IMREAD_COLOR)
#         return img
#     except:
#         return None

# def compute_avg_flow(img1, img2):
#     gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#     gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#     flow = cv2.calcOpticalFlowFarneback(
#         gray1, gray2, None,
#         pyr_scale=0.5, levels=3, winsize=15,
#         iterations=3, poly_n=5, poly_sigma=1.2, flags=0
#     )
#     mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
#     return np.mean(mag)


# # ---------- Configuration ----------

# tar_path = "../rolling-shutter-data/10-20-final.tar"
# methodA_dir = "output/intermediate_10_20_val/depth_npy"
# methodB_dir = "output/RS-diff-finetuned"

# # ---------- Load Ground Truth Tar ----------
# tar = tarfile.open(tar_path, "r")
# tar_members = tar.getnames()

# # Collect all output images from one method
# out_list = sorted([f for f in glob(os.path.join(methodA_dir, "**", "*.png"), recursive=True) if is_image_file(f)])

# flows, flow_percent, psnr_A, psnr_B = [], [], [], []

# for outA_path in tqdm(out_list):
#     rel_path = os.path.relpath(outA_path, methodA_dir)
#     # print(rel_path)
#     # Build matching paths for GT and input inside tar
#     gt_rel_path = rel_path.replace("camera1", "camera1")  # GT path as-is
#     input_rel_path = rel_path.replace("camera1", "camera2")  # Input path

#     gt_match = next((m for m in tar_members if m.endswith(gt_rel_path)), None)
#     input_match = next((m for m in tar_members if m.endswith(input_rel_path)), None)
#     # print(gt_match, input_match)
#     if gt_match is None or input_match is None:
#         continue

#     # Find corresponding method B output
#     outB_path = os.path.join(methodB_dir, rel_path.replace(".png", ".png.jpg").replace("camera1", "camera2"))
#     # print(outB_path)
#     if not os.path.exists(outB_path):
#         continue

#     try:
#         gt = read_tar_image(tar, gt_match)
#         inp = read_tar_image(tar, input_match)
#         outA = cv2.imread(outA_path, cv2.IMREAD_COLOR)
#         outB = cv2.imread(outB_path, cv2.IMREAD_COLOR)

#         if gt is None or inp is None or outA is None or outB is None:
#             continue

#         # Resize to match
#         h, w = gt.shape[:2]
#         outA = cv2.resize(outA, (w, h))
#         outB = cv2.resize(outB, (w, h))
#         inp = cv2.resize(inp, (w, h))

#         # Compute metrics
#         avg_flow = compute_avg_flow(inp, gt)
#         flow_percent_s = (avg_flow / (h**2 + w**2)**1/2)*100
#         psnr_a = PSNR(outA, gt)
#         psnr_b = PSNR(outB, gt)

#         flows.append(avg_flow)
#         flow_percent.append(flow_percent_s)
#         psnr_A.append(psnr_a)
#         psnr_B.append(psnr_b)
#     except Exception as e:
#         print(f"Error on {rel_path}: {e}")
#         continue

# tar.close()


# # ---------- Plot Results ----------
# mpl.rcParams.update({
#     'font.size': 20,          # base font size
#     'axes.labelsize': 18,     # x/y label size
#     'axes.titlesize': 18,     # title size
#     'xtick.labelsize': 18,    # x-tick size
#     'ytick.labelsize': 18,    # y-tick size
#     'legend.fontsize': 18,    # legend text size
#     'figure.titlesize': 20,   # overall figure title size
# })

# flows = np.array(flows)
# psnr_A = np.array(psnr_A)
# psnr_B = np.array(psnr_B)
# diff = psnr_A - psnr_B

# plt.figure(figsize=(8,6))
# plt.scatter(flows, psnr_A, label='Method A', alpha=0.7)
# plt.scatter(flows, psnr_B, label='Method B', alpha=0.7)
# plt.xlabel("Average Optical Flow Magnitude (camera2 → camera1)")
# plt.ylabel("PSNR")
# plt.title("PSNR vs. Optical Flow Magnitude")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("psnr_vs_flow.png", dpi=200)

# plt.figure(figsize=(8,4))
# plt.scatter(flows, diff, color='purple', alpha=0.7)
# plt.axhline(0, color='gray', linestyle='--')
# plt.xlabel("Average Optical Flow Magnitude (camera2 → camera1)")
# plt.ylabel("ΔPSNR (Ours - RS-Diff)")
# plt.title("PSNR Difference vs. Optical Flow")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("psnr_diff_vs_flow.png", dpi=200)
# plt.show()

# print("✅ Done! Saved plots: psnr_vs_flow.png and psnr_diff_vs_flow.png")
# # ---------- Binning & Line Plot (Zoomed Y-Axis) ----------

# # Define bins (e.g., 0-5, 5-10, ...)
# bin_edges = np.arange(0, np.ceil(np.array(flows).max()) + 5, 5)

# bin_means = []
# percent_a_better = []

# for i in range(len(bin_edges)-1):
#     mask = (flows >= bin_edges[i]) & (flows < bin_edges[i+1])
#     if np.sum(mask) == 0:
#         continue
#     bin_mean = np.mean(flows[mask])
#     if bin_mean > 20:
#         continue
#     percent = np.sum(psnr_A[mask] > psnr_B[mask]) / np.sum(mask) * 100
#     bin_means.append(bin_mean)
#     percent_a_better.append(percent)

# # Plot as a connected line graph
# plt.figure(figsize=(8,4))
# plt.plot(bin_means, percent_a_better, marker='o', linestyle='-', color='teal')
# plt.fill_between(bin_means, percent_a_better, alpha=0.2, color='teal')

# plt.xlabel("Average Optical Flow Magnitude")
# plt.ylabel("Ours > RS-Diff (%)")
# plt.title("Cross Comparison Across Optical Flow Magnitudes")
# plt.ylim(65, 100)  # 👈 Zoomed y-axis range
# plt.xlim(0, 18)
# plt.grid(True, linestyle='--', alpha=0.7)
# # plt.legend()
# plt.tight_layout()
# plt.savefig("psnr_bin_line_vs_flow_zoomed.png", dpi=200)
# plt.show()

# print("✅ Done! Saved plot: psnr_bin_line_vs_flow_zoomed.png")

# print("Our PSNR: ", np.mean(psnr_A))
# print("RS-diff-finetuned PSNR: ", np.mean(psnr_B))

import os
import cv2
import tarfile
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from math import log10
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim  # new import

# ---------- Helper Functions ----------

def is_image_file(path):
    return path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))

def PSNR(img1, img2):
    img1 = img1.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * log10(1.0 / mse)

def compute_ssim(img1, img2):
    # Convert to grayscale for SSIM
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return ssim(gray1, gray2, data_range=gray2.max() - gray2.min())

def read_tar_image(tar, member_name):
    try:
        f = tar.extractfile(member_name)
        if f is None:
            return None
        data = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except:
        return None

def compute_avg_flow(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    return np.mean(mag)


# ---------- Configuration ----------

tar_path = "../rolling-shutter-data/10-20-final.tar"
methodA_dir = "output/intermediate_10_20_val/depth_npy"
methodB_dir = "output/RS-diff"

# ---------- Load Ground Truth Tar ----------
tar = tarfile.open(tar_path, "r")
tar_members = tar.getnames()

# Collect all output images from one method
out_list = sorted([f for f in glob(os.path.join(methodA_dir, "**", "*.png"), recursive=True) if is_image_file(f)])

flows, flow_percent, psnr_A, psnr_B, ssim_A, ssim_B = [], [], [], [], [], []

for outA_path in tqdm(out_list):
    rel_path = os.path.relpath(outA_path, methodA_dir)
    gt_rel_path = rel_path.replace("camera1", "camera1")
    input_rel_path = rel_path.replace("camera1", "camera2")

    gt_match = next((m for m in tar_members if m.endswith(gt_rel_path)), None)
    input_match = next((m for m in tar_members if m.endswith(input_rel_path)), None)
    if gt_match is None or input_match is None:
        continue

    outB_path = os.path.join(methodB_dir, rel_path.replace(".png", ".png.jpg").replace("camera1", "camera2"))
    if not os.path.exists(outB_path):
        continue

    try:
        gt = read_tar_image(tar, gt_match)
        inp = read_tar_image(tar, input_match)
        outA = cv2.imread(outA_path, cv2.IMREAD_COLOR)
        outB = cv2.imread(outB_path, cv2.IMREAD_COLOR)

        if gt is None or inp is None or outA is None or outB is None:
            continue

        h, w = gt.shape[:2]
        outA = cv2.resize(outA, (w, h))
        outB = cv2.resize(outB, (w, h))
        inp = cv2.resize(inp, (w, h))

        avg_flow = compute_avg_flow(inp, gt)
        flow_percent_s = (avg_flow / (h**2 + w**2)**0.5) * 100
        psnr_a = PSNR(outA, gt)
        psnr_b = PSNR(outB, gt)
        ssim_a = compute_ssim(outA, gt)
        ssim_b = compute_ssim(outB, gt)

        flows.append(avg_flow)
        flow_percent.append(flow_percent_s)
        psnr_A.append(psnr_a)
        psnr_B.append(psnr_b)
        ssim_A.append(ssim_a)
        ssim_B.append(ssim_b)

    except Exception as e:
        print(f"Error on {rel_path}: {e}")
        continue

tar.close()

# ---------- Convert to arrays ----------
flows = np.array(flows)
flow_percent = np.array(flow_percent)
psnr_A = np.array(psnr_A)
psnr_B = np.array(psnr_B)
ssim_A = np.array(ssim_A)
ssim_B = np.array(ssim_B)

# ---------- Print mean metrics ----------
print("===== Mean Metrics =====")
print(f"Method A PSNR: {np.mean(psnr_A):.3f}")
print(f"Method B PSNR: {np.mean(psnr_B):.3f}")
print(f"Method A SSIM: {np.mean(ssim_A):.4f}")
print(f"Method B SSIM: {np.mean(ssim_B):.4f}")
