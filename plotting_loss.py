from src.util.loss import get_loss
import tarfile
from PIL import Image
import numpy as np
import torch
from marigold.derolling_pipeline import MarigoldRGBPipeline
import os
import matplotlib.pyplot as plt

def MeanAbsRelLoss(rs, gs):
    diff = rs - gs
    rel_abs = abs(diff/gs)
    return rel_abs.mean()

losses = ["silog_mse", "silog_rmse", "mse_loss", "l1,loss", "mean_abs_rel"]

def extract_image_from_tar(tar_file_path, image_name, destination_folder):
    """Extracts a specific image from a tar archive.

    Args:
        tar_file_path: Path to the tar archive file.
        image_name: Name of the image file to extract.
        destination_folder: Folder where the image should be extracted.
    """

    with tarfile.open(tar_file_path, "r") as tar:
        for member in tar.getmembers():
            if member.name == image_name:
                tar.extract(member, destination_folder)
                print(f"Image '{image_name}' extracted to '{destination_folder}'")
                return
        print(f"Image '{image_name}' not found in the tar archive.")

def encode_rgb(rgb, vae):
    rgb = np.transpose(rgb, (2, 0, 1))
    tensor = torch.from_numpy(rgb)
    rgb_latent_scale_factor = 0.18215

    h = vae.encoder(torch.unsqueeze(tensor.float(), 0))
    moments = vae.quant_conv(h)
    mean, logvar = torch.chunk(moments, 2, dim=1)
    # scale latent
    latent = mean * rgb_latent_scale_factor
    return latent

# Example usage:
tar_file_path = "../rolling-shutter-data/fan_diverse.tar"
image_names = ["RS0.05/", "RS0.1/", "RS0.2/", "RS0.3/", "RS0.4/", "RS0.5/"]
shutter_val = [0.05, 0.1,  0.2, 0.3, 0.4, 0.5]
destination_folder = "output/loss/extracted_images"
# for image in image_names:
# for i in range(10):
#     extract_image_from_tar(tar_file_path, "table_fan/GS/" + "000" + str(i+1) + ".png", destination_folder)




model = MarigoldRGBPipeline.from_pretrained(
    os.path.join("../stable-diffusion-2")
)
vae = model.vae
# encode
i = 0
num_examples = 9
val = np.empty(len(image_names))

y = np.zeros((num_examples,  len(image_names)))
x = shutter_val
loss = get_loss("mse_loss")
parent_dir = "input/centered/"
for k in range(num_examples):
    GS = Image.open(parent_dir + "GS/000" + str(k+1) + ".png")
    GS_np = np.array(GS)
    global_latent = encode_rgb(GS_np, vae)
    i = 0
    for image in image_names:
        RS = Image.open(parent_dir + image + "000" + str(k+1) + ".png")
        RS_np = np.array(RS)
        RS_latent = encode_rgb(RS_np, vae)
        y[k, i] = loss(RS_latent.float(), global_latent.float()).mean().item()
        i+=1
colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'black', 'brown', 'pink']
for  i in range(num_examples):
    plt.scatter(x, y[i], marker='o', color=colors[i%len(colors)])
plt.title("Loss as a Function of Rolling Shutter for the Centered Data")
plt.ylabel("Mean Squared Error (Global - Rolling)")
plt.xlabel("Rolling Shutter Amount")
plt.savefig("output/loss/plot_centered" + ".png")







