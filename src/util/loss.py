# Author: Bingxin Ke
# Last modified: 2024-02-22

import torch
import numpy as np
import pickle

import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn import functional as F
from src.util.DISTS import DISTS as DISTS_Model


def get_loss(loss_name, **kwargs):
    if "silog_mse" == loss_name:
        criterion = SILogMSELoss(**kwargs)
    elif "silog_rmse" == loss_name:
        criterion = SILogRMSELoss(**kwargs)
    elif "mse_loss" == loss_name:
        decoder = kwargs.pop("decoder", None)
        criterion = torch.nn.MSELoss(**kwargs)
    elif "l1_loss" == loss_name:
        criterion = torch.nn.L1Loss(**kwargs)
    elif "l1_loss_with_mask" == loss_name:
        criterion = L1LossWithMask(**kwargs)
    elif "mean_abs_rel" == loss_name:
        criterion = MeanAbsRelLoss()
    elif "perceptual_loss" == loss_name:
        criterion = PerceptualLoss(vae=decoder, **kwargs)
    elif "combined_loss" == loss_name:
        decoder = kwargs.pop("decoder", None)
        if decoder is None:
            raise ValueError("decoder must be provided for perceptual_loss")
        criterion = CombinedLoss(decoder=decoder, **kwargs)
    elif "combined_dists" == loss_name:
        decoder = kwargs.pop("decoder", None)
        if decoder is None:
            raise ValueError("decoder must be provided for perceptual_dists")
        criterion = CombinedDISTS(decoder=decoder, **kwargs)
    elif "combined_mse" == loss_name:
        decoder = kwargs.pop("decoder", None)
        if decoder is None:
            raise ValueError("decoder must be provided for perceptual_dists")
        criterion = CombinedMSE(decoder=decoder, **kwargs)
    else:
        raise NotImplementedError

    return criterion


class L1LossWithMask:
    def __init__(self, batch_reduction=False):
        self.batch_reduction = batch_reduction

    def __call__(self, depth_pred, depth_gt, valid_mask=None):
        diff = depth_pred - depth_gt
        if valid_mask is not None:
            diff[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
        else:
            n = depth_gt.shape[-2] * depth_gt.shape[-1]

        loss = torch.sum(torch.abs(diff)) / n
        if self.batch_reduction:
            loss = loss.mean()
        return loss

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

    distances = np.sqrt(np.sum(new_points**2 * inv_cov, axis=1))  # (num_samples,)

    return np.mean(distances)  # Return both individual and mean distances

class MeanAbsRelLoss:
    def __init__(self) -> None:
        # super().__init__()
        with open("pca_model_2.pkl", "rb") as f:
            self.pca = pickle.load(f)
        
        pass

    def __call__(self, pred, gt):
        diff = pred - gt
        rel_abs = torch.abs(diff / gt)
        loss_mean = torch.mean(rel_abs, dim=0)

        # explained_variance = (self.pca.explained_variance_)
        # latents_np = pred.cpu().detach().numpy()
        # num_samples, channels, height, width = latents_np.shape
        # reshaped_latents = latents_np.reshape(num_samples, -1)
        # new_point = self.pca.transform(reshaped_latents)
        # # Compute Mahalanobis distance
        # distance = mahalanobis_distance_batch(new_point, explained_variance)
        # print("Distance")
        # print(distance)
        # print(loss_mean.mea/n())

        #ratio of loss!!!!
        return loss_mean # + 1/2*distance


class SILogMSELoss:
    def __init__(self, lamb=1, log_pred=True, batch_reduction=True):
        """Scale Invariant Log MSE Loss

        Args:
            lamb (_type_): lambda, lambda=1 -> scale invariant, lambda=0 -> L2 loss
            log_pred (bool, optional): True if model prediction is logarithmic depht. Will not do log for depth_pred
        """
        super(SILogMSELoss, self).__init__()
        self.lamb = lamb
        self.pred_in_log = log_pred
        self.batch_reduction = batch_reduction

    def __call__(self, depth_pred, depth_gt, valid_mask=None):
        log_depth_pred = (
            depth_pred if self.pred_in_log else torch.log(torch.clip(depth_pred, 1e-8))
        )
        log_depth_gt = torch.log(depth_gt)

        diff = log_depth_pred - log_depth_gt
        if valid_mask is not None:
            diff[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
        else:
            n = depth_gt.shape[-2] * depth_gt.shape[-1]

        diff2 = torch.pow(diff, 2)

        first_term = torch.sum(diff2, (-1, -2)) / n
        second_term = self.lamb * torch.pow(torch.sum(diff, (-1, -2)), 2) / (n**2)
        loss = first_term - second_term
        if self.batch_reduction:
            loss = loss.mean()
        return loss


class SILogRMSELoss:
    def __init__(self, lamb, alpha, log_pred=True):
        """Scale Invariant Log RMSE Loss

        Args:
            lamb (_type_): lambda, lambda=1 -> scale invariant, lambda=0 -> L2 loss
            alpha:
            log_pred (bool, optional): True if model prediction is logarithmic depht. Will not do log for depth_pred
        """
        super(SILogRMSELoss, self).__init__()
        self.lamb = lamb
        self.alpha = alpha
        self.pred_in_log = log_pred

    def __call__(self, depth_pred, depth_gt, valid_mask):
        log_depth_pred = depth_pred if self.pred_in_log else torch.log(depth_pred)
        log_depth_gt = torch.log(depth_gt)
        # borrowed from https://github.com/aliyun/NeWCRFs
        # diff = log_depth_pred[valid_mask] - log_depth_gt[valid_mask]
        # return torch.sqrt((diff ** 2).mean() - self.lamb * (diff.mean() ** 2)) * self.alpha

        diff = log_depth_pred - log_depth_gt
        if valid_mask is not None:
            diff[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
        else:
            n = depth_gt.shape[-2] * depth_gt.shape[-1]

        diff2 = torch.pow(diff, 2)
        first_term = torch.sum(diff2, (-1, -2)) / n
        second_term = self.lamb * torch.pow(torch.sum(diff, (-1, -2)), 2) / (n**2)
        loss = torch.sqrt(first_term - second_term).mean() * self.alpha
        return loss
class Loss(torch.nn.Module):
    def __init__(self, vae=None):
        self.depth_latent_scale_factor = 0.18215
        super(Loss, self, ).__init__()
        self.vae = vae

    def decode_rgb(self, global_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (`torch.Tensor`):
                Depth latent to be decoded.

        Returns:
            `torch.Ten
            sor`: Decoded depth map.
        """
        self.vae.eval()
        with torch.no_grad():
        # scale latent
            global_latent = global_latent / self.depth_latent_scale_factor
            # decode
            z = self.vae.post_quant_conv(global_latent)
            stacked = self.vae.decoder(z)
            # mean of output channels
            global_mean = stacked
        return global_mean
    
class PerceptualLoss(Loss):
    def __init__(self, layers=('relu1_2',), resize=True, vae=None):
        super(PerceptualLoss, self).__init__(vae = vae)
        self.vgg = models.vgg16(pretrained=True).features.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.layer_map = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_3': 15,
            'relu4_3': 22,
            'relu5_3': 29
        }

        self.selected_layers = [self.layer_map[layer] for layer in layers]
        self.resize = resize
        self.transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.depth_latent_scale_factor = 0.18215
        self.vae = vae

    def forward(self, pred, gt):
        # Convert 1-channel to 3-channel if needed
        pred_rgb = self.decode_rgb(pred)
        gt_rgb = self.decode_rgb(gt)
        if pred_rgb.shape[1] == 1:
            pred_rgb = pred_rgb.repeat(1, 3, 1, 1)
            gt_rgb = gt_rgb.repeat(1, 3, 1, 1)

        # Normalize
        pred_rgb = self.transform(pred_rgb)
        gt_rgb = self.transform(gt_rgb)

        # Resize to 224x224 for VGG input if needed
        if self.resize:
            pred_rgb = F.interpolate(pred_rgb, size=(224, 224), mode='bilinear', align_corners=False)
            gt_rgb = F.interpolate(gt_rgb, size=(224, 224), mode='bilinear', align_corners=False)

        loss = 0.0
        x = pred_rgb
        y = gt_rgb
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)
            if i in self.selected_layers:
                loss += F.l1_loss(x, y)

        return loss
    
class DISTS(Loss):
    def __init__(self, vae):
        super(DISTS, self).__init__(vae = vae)
        self.model = DISTS_Model().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def forward(self, pred, gt, **kwargs):
        pred_rgb = self.decode_rgb(pred)
        gt_rgb = self.decode_rgb(gt)

        return self.model(pred_rgb, gt_rgb, **kwargs)

class CombinedMSE(Loss):
    def __init__(self, decoder, alpha=1, beta=1):
        super(CombinedMSE, self).__init__(vae = decoder)
        self.l1 = torch.nn.MSELoss(reduction="mean")
        self.alpha = alpha
        self.beta = beta
    def forward(self, pred, gt, **kwargs):
        pred_rgb = self.decode_rgb(pred)
        gt_rgb = self.decode_rgb(gt)

        return self.alpha*self.l1(pred,gt) + self.beta*self.l1(pred_rgb, gt_rgb)

    
class CombinedLoss(torch.nn.Module):
    def __init__(self, decoder, alpha=1, beta=0.08):
        super(CombinedLoss, self).__init__()
        self.l1 = torch.nn.MSELoss(reduction="mean")
        self.perceptual = PerceptualLoss(vae=decoder)
        self.perceptual = self.perceptual.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, gt):
        mse = self.alpha * self.l1(pred, gt)
        perc = self.beta* self.perceptual(pred, gt)
        # print(f"Loss Split: {mse.mean()} : {perc}")
        return mse + perc

class CombinedDISTS(torch.nn.Module):
    def __init__(self, decoder, alpha=1, beta = 1):
        super(CombinedDISTS, self).__init__()
        self.l1 = torch.nn.MSELoss(reduction="mean")
        self.dists = DISTS(vae = decoder)
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, gt):
        mse = self.alpha * self.l1(pred, gt)
        dists = self.beta*self.dists(pred,gt, require_grad=True, batch_average=True)
        # print(f"Loss {mse}:{dists}")
        return mse + dists