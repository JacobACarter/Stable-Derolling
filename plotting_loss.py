from src.util.loss import get_loss
import tarfile
from PIL import Image
import numpy as np
import torch
from marigold.derolling_pipeline import MarigoldRGBPipeline

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

# Example usage:
# tar_file_path = "../rolling-shutter-data/ceiling_fan.tar"
# image_names = ["ceiling_fan/RS0.05/0001.png", "ceiling_fan/RS0.1/0001.png", "ceiling_fan/RS0.15/0001.png",
#     "ceiling_fan/RS0.2/0001.png", "ceiling_fan/RS0.25/0001.png", "ceiling_fan/RS0.3/0001.png", "ceiling_fan/RS0.4/0001.png", "ceiling_fan/RS0.5/0001.png"]
# destination_folder = "output/loss/extracted_images"
# for image in image_names:
#     extract_image_from_tar(tar_file_path, image, destination_folder)


# GS = Image.open("output/loss/extracted_images/ceiling_fan/GS/0001.png")
# GS_np = np.array(GS)

# for image in image_names:
#     RS = Image.open("output/loss/extracted_images/" + image)
#     RS_np = np.array(RS)
#     loss = MeanAbsRelLoss(RS_np, GS_np)
#     print(image + ": " +str(loss))

model = MarigoldRGBPipeline.from_pretrained(
    os.path.join("../stable-diffusion-2")
)



