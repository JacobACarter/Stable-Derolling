import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import vgg16
from torchvision.datasets import ImageFolder
import torch.nn.functional as F

from diffusers import StableDiffusionPipeline  # or your custom pipeline

# ----------------------------
# 1. Load your pipeline
# ----------------------------

pipeline = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4"
).to("cuda")

vae = pipeline.vae  # your encoder/decoder
vae.train()  # Make sure in train mode

# ----------------------------
# 2. Freeze UNet
# ----------------------------

for param in pipeline.unet.parameters():
    param.requires_grad = False

# ----------------------------
# 3. Optional: freeze text encoder if needed
# ----------------------------

if hasattr(pipeline, "text_encoder"):
    for param in pipeline.text_encoder.parameters():
        param.requires_grad = False

# ----------------------------
# 4. Make your dataset + loader
# ----------------------------

transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),  # SD expects [-1, 1]
])

dataset = ImageFolder("path/to/images", transform=transform)
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

# ----------------------------
# 5. Optional: VGG perceptual loss
# ----------------------------

vgg = vgg16(pretrained=True).features[:16].eval().cuda()
for p in vgg.parameters():
    p.requires_grad = False

def perceptual_loss(x, y):
    return F.l1_loss(vgg(x), vgg(y))

# ----------------------------
# 6. Optimizer for VAE only
# ----------------------------

optimizer = torch.optim.AdamW(vae.parameters(), lr=1e-5)

# ----------------------------
# 7. Training loop
# ----------------------------

num_epochs = 5

for epoch in range(num_epochs):
    for batch in loader:
        real_images, _ = batch
        real_images = real_images.cuda()

        # 1) Encode → decode
        posterior = vae.encode(real_images).latent_dist
        latents = posterior.sample()
        latents = latents * 0.18215  # keep SD scaling factor

        recon_images = vae.decode(latents).sample  # [B, 3, H, W]

        # 2) Reconstruction loss
        recon_loss = F.l1_loss(recon_images, real_images)

        # 3) Perceptual loss
        p_loss = perceptual_loss(recon_images, real_images)

        # 4) KL divergence loss
        mu, logvar = posterior.mean, posterior.logvar
        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1))

        total_loss = recon_loss + 0.1 * p_loss + 0.01 * kl_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"Loss: {total_loss.item():.4f} | Recon: {recon_loss.item():.4f} | KL: {kl_loss.item():.4f}")

    torch.save(vae.state_dict(), f"vae_epoch_{epoch}.pt")

print("✅ Done fine-tuning VAE!")
