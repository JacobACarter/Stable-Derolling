import argparse
import logging
import os
import shutil
from datetime import datetime, timedelta
from typing import List
import pickle as pkl
import torch
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
import numpy as np
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import random
import time

from marigold.derolling_pipeline import MarigoldRGBPipeline
from src.dataset import BaseRGBDataset, DatasetMode, get_dataset
from src.dataset.mixed_sampler import MixedBatchSampler
from src.trainer import get_trainer_cls
from PIL import Image
from io import BytesIO
from sklearn.decomposition import PCA
from torchvision import transforms
from tqdm import tqdm
import pickle
import numpy as np
from scipy.spatial.distance import mahalanobis
from src.util.config_util import (
    find_value_in_omegaconf,
    recursive_load_config,
)
from src.util.depth_transform import (
    DepthNormalizerBase,
    get_depth_normalizer,
)
from src.util.logging_util import (
    config_logging,
    init_wandb,
    load_wandb_job_id,
    log_slurm_job_id,
    save_wandb_job_id,
    tb_logger,
)
from src.util.slurm_util import get_local_scratch_dir, is_on_slurm

from torch.nn import Conv2d
from torch.nn.parameter import Parameter
import zipfile

def image_generator(zip_ref, batch_size, city, num_samples):

        
        image_filenames = [f for f in zip_ref.namelist() if (f.lower().endswith(('.png', '.jpg', '.jpeg')) and city in f.lower())]
        if num_samples < len(image_filenames):
            image_filenames = random.sample(image_filenames, k=num_samples)
        for i in range(0, len(image_filenames), batch_size):
            batch_images = []
            batch_filenames = image_filenames[i:i + batch_size]

            for filename in batch_filenames:
                with zip_ref.open(filename) as file:
                    image = Image.open(BytesIO(file.read())).convert("RGB")  # Convert to RGB
                    batch_images.append(transform(image).unsqueeze(0))  # Convert to tensor and add batch dimension
            
            yield torch.cat(batch_images, dim=0).to(device)  # Return batch


def _replace_unet_conv_in(model):
    # replace the first layer to accept 8 in_channels
    _weight = model.unet.conv_in.weight.clone()  # [320, 4, 3, 3]
    _bias = model.unet.conv_in.bias.clone()  # [320]
    _weight = _weight.repeat((1, 2, 1, 1))  # Keep selected channel(s)
    # half the activation magnitude
    _weight *= 0.5
    # new conv_in channel
    _n_convin_out_channel = model.unet.conv_in.out_channels
    _new_conv_in = Conv2d(
        8, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
    )
    _new_conv_in.weight = Parameter(_weight)
    _new_conv_in.bias = Parameter(_bias)
    model.unet.conv_in = _new_conv_in
    logging.info("Unet conv_in layer is replaced")
    # replace config
    model.unet.config["in_channels"] = 8
    logging.info("Unet config is updated")
    return

#Set environmental variables
gpu_num = 5

os.environ["BASE_DATA_DIR"] = "../rolling-shutter-data"
os.environ["BASE_CKPT_DIR"] = "../"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
print("argument gpu: " + str(gpu_num) + "\n")
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)

#load device:
# -------------------- Device --------------------

if torch.cuda.is_available():
    device = torch.device("cuda")

logging.info(f"device = {device}")


# Load your trained AutoencoderKL model
pipe: MarigoldRGBPipeline = MarigoldRGBPipeline.from_pretrained(
    "../stable-diffusion-2"
)
if 8 != pipe.unet.config["in_channels"]:
    _replace_unet_conv_in(pipe)

model_path = "output/train_X4K_end_1/checkpoint/latest/unet/diffusion_pytorch_model.bin"
pipe.unet.load_state_dict(
    torch.load(model_path, map_location=device)

)
pipe.unet.to(device)
logging.info(f"loaded unet parameters from {model_path}")


encoder = pipe.vae
encoder.to(device)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((300,300)),
  ])



# Load PCA analysis from pickle file
with open("pca_model_2.pkl", "rb") as f:
    pca = pickle.load(f)

# Extract PCA components and mean
mean = (pca.mean_ )  # Mean of the training data
components = (pca.components_ ) # Principal components
explained_variance = (pca.explained_variance_)  # Variance along each principal component

def mahalanobis_distance_batch(new_points,explained_variance):
    """
    Computes Mahalanobis distance for each point in a batch.

    Parameters:
    - new_points: (num_samples, n_components) transformed PCA vectors
    - mean: (n_components,) PCA-space mean
    - components: (n_components, original_dim) PCA components
    - explained_variance: (n_components,) variance along each PCA axis

    Returns:
    - distances: (num_samples,) array of Mahalanobis distances
    - mean_distance: scalar mean Mahalanobis distance across batch
    """
    # Ensure inputs are numpy arrays
    new_points = np.asarray(new_points)  # (num_samples, n_components)



    inv_cov = 1 / explained_variance  # (n_components,)

    distances = np.sqrt(np.sum(new_point**2 * inv_cov, axis=1))  # (num_samples,)

    return distances, np.mean(distances)  # Return both individual and mean distances

batch_images = []
# Example new point
image = Image.open("./input/X4K/occ008.894_f2881.png").convert("RGB")  # Convert to RGB
new_size = (300, 300)
image = image.resize(new_size)
image = np.asarray(image)
image = np.transpose(image, (2, 0, 1)).astype(int)
image = image / 255.0 * 2.0 - 1.0 
batch_images.append(torch.from_numpy(image).unsqueeze(0).float())  # Convert to tensor and add batch dimension
image_batch = torch.cat(batch_images, dim=0).to(device) 

latent_vectors = []
with torch.no_grad():
    # st = time.time()
    encoded = pipe.encode_rgb(image_batch)
        # Extract mean of latent distribution
    latent_vectors.append(encoded.cpu().numpy())  # Move to CPU to save memory

# Convert list of arrays to a single numpy array
latents_np = np.concatenate(latent_vectors, axis=0)
num_samples, channels, height, width = latents_np.shape
reshaped_latents = latents_np.reshape(num_samples, -1)
new_point = pca.transform(reshaped_latents)
# Compute Mahalanobis distance
distance = mahalanobis_distance_batch(new_point, explained_variance)
print("Mahalanobis Distance:", distance)