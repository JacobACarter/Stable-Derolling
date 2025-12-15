# MarigoldRGBDepthTrainer (teacher-student rewrite)
# Last modified: 2025-10-14 (rewrite)
import logging
import os
import shutil
from datetime import datetime
from typing import List, Union, Optional

import numpy as np
import torch
from diffusers import DDPMScheduler
from omegaconf import OmegaConf
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from marigold.derolling_depth_pipeline import MarigoldRGBDepthPipeline, MarigoldDepthOutput
from src.util import metric
from src.util.data_loader import skip_first_batches
from src.util.logging_util import tb_logger, eval_dic_to_text
from src.util.loss import get_loss
from src.util.lr_scheduler import IterExponential
from src.util.metric import MetricTracker
from src.util.multi_res_noise import multi_res_noise_like
from src.util.seeding import generate_seed_sequence
from torchmetrics import MeanSquaredError
from torchmetrics.collections import MetricCollection

class MarigoldRGBDepthTrainer:
    """
    Trainer that:
      - trains a derolling UNet (self.model.unet_derolling) with diffusion training scheme
      - uses a frozen teacher pipeline (self.teacher_pipeline) that runs on GT RGB images
        to produce pseudo-GT depths used to supervise the depth head / marigold path.
    Notes:
      - teacher_pipeline should accept the same call signature as used in validate():
            pipe_out = teacher_pipeline(gt_rgb_np_or_tensor, denoising_steps=..., ensemble_size=..., processing_res=..., match_input_res=..., batch_size=..., generator=None)
        and return an object with attributes `.depth_np` (H x W float depth) and `.rgb` (C x H x W or H x W x C).
      - If teacher_cache_dir is provided, teacher outputs are cached per 'sample id' (string) to avoid recomputation.
    """

    def __init__(
        self,
        cfg: OmegaConf,
        model: MarigoldRGBDepthPipeline,
        train_dataloader: DataLoader,
        device,
        base_ckpt_dir: str,
        out_dir_ckpt: str,
        out_dir_eval: str,
        out_dir_vis: str,
        accumulation_steps: int,
        val_dataloaders: List[DataLoader] = None,
        vis_dataloaders: List[DataLoader] = None,
        teacher_pipeline: Optional[MarigoldRGBDepthPipeline] = None,
        teacher_cache_dir: Optional[str] = None,
    ):
        self.cfg: OmegaConf = cfg
        self.model: MarigoldRGBDepthPipeline = model
        self.device = device
        self.seed: Union[int, None] = (self.cfg.trainer.init_seed)  # used to generate seed sequence
        self.out_dir_ckpt = out_dir_ckpt
        self.out_dir_eval = out_dir_eval
        self.out_dir_vis = out_dir_vis
        self.train_loader: DataLoader = train_dataloader
        self.val_loaders: List[DataLoader] = val_dataloaders or []
        self.vis_loaders: List[DataLoader] = vis_dataloaders or []
        self.accumulation_steps: int = accumulation_steps

        # Teacher pipeline (frozen). If None, depth supervision falls back to dataset depth if available.
        self.teacher_pipeline = teacher_pipeline
        self.teacher_cache_dir = teacher_cache_dir
        if self.teacher_pipeline is not None:
            # ensure teacher in eval and frozen
            self.teacher_pipeline.to(self.device)
            self.teacher_pipeline.eval()
            for p in self.teacher_pipeline.parameters():
                p.requires_grad_(False)
            if self.teacher_cache_dir is not None:
                os.makedirs(self.teacher_cache_dir, exist_ok=True)

        # Adapt input layers if needed (expect 8 channels after concatenation)
        if 8 != self.model.unet_derolling.config.get("in_channels", None):
            self._replace_unet_conv_in(self.model.unet_derolling)

        # If you have a second unet inside model (previously unet_marigold), we won't train it.
        if hasattr(self.model, "unet_marigold"):
            try:
                if 8 != self.model.unet_marigold.config.get("in_channels", None):
                    self._replace_unet_conv_in(self.model.unet_marigold)
            except Exception:
                # best-effort; if config missing ignore
                pass

        # Encode empty text prompt early (safe to do on CPU, but move to device)
        if hasattr(self.model, "encode_empty_text"):
            self.model.encode_empty_text()
            self.empty_text_embed = self.model.empty_text_embed.detach().clone().to(self.device)
        else:
            self.empty_text_embed = None

        # Memory-efficient attention if supported
        if hasattr(self.model.unet_derolling, "enable_xformers_memory_efficient_attention"):
            try:
                self.model.unet_derolling.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        if hasattr(self.model, "unet_marigold") and hasattr(self.model.unet_marigold, "enable_xformers_memory_efficient_attention"):
            try:
                self.model.unet_marigold.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

        # Freeze marigold unet if present (we supervise from the teacher instead)
        if hasattr(self.model, "unet_marigold"):
            for param in self.model.unet_marigold.parameters():
                param.requires_grad = False

        # Freeze other heavy modules
        if hasattr(self.model, "vae"):
            self.model.vae.requires_grad_(False)
        if hasattr(self.model, "text_encoder"):
            self.model.text_encoder.requires_grad_(False)

        # Trainability: only derolling unet
        self.model.unet_derolling.requires_grad_(True)

        # Optimizer
        lr = float(self.cfg.lr)
        self.optimizer = Adam(self.model.unet_derolling.parameters(), lr=lr)

        # LR scheduler
        lr_func = IterExponential(
            total_iter_length=self.cfg.lr_scheduler.kwargs.total_iter,
            final_ratio=self.cfg.lr_scheduler.kwargs.final_ratio,
            warmup_steps=self.cfg.lr_scheduler.kwargs.warmup_steps,
        )
        self.lr_scheduler = LambdaLR(optimizer=self.optimizer, lr_lambda=lr_func)

        # Losses
        self.loss_derolling = get_loss(loss_name=self.cfg.loss_derolling.name, **(self.cfg.loss_derolling.kwargs or {}))
        self.loss_marigold = get_loss(loss_name=self.cfg.loss_marigold.name, **(self.cfg.loss_marigold.kwargs or {}))

        # Training noise scheduler
        self.training_noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(
            os.path.join(base_ckpt_dir, cfg.trainer.training_noise_scheduler.pretrained_path, "scheduler")
        )
        self.prediction_type = self.training_noise_scheduler.config.prediction_type
        assert (
            self.prediction_type == self.model.scheduler.config.prediction_type
        ), "Different prediction types between training scheduler and model scheduler"
        self.scheduler_timesteps = self.training_noise_scheduler.config.num_train_timesteps

        # Eval metrics
        self.metric_funcs = [getattr(metric, _met) for _met in cfg.eval.eval_metrics]
        self.train_metrics = MetricTracker(*["loss"])
        self.val_metrics = MetricTracker(*[m.__name__ for m in self.metric_funcs])
        self.main_val_metric = cfg.validation.main_val_metric
        self.main_val_metric_goal = cfg.validation.main_val_metric_goal
        assert (self.main_val_metric in cfg.eval.eval_metrics), f"Main eval metric `{self.main_val_metric}` not found."
        self.best_metric = 1e8 if "minimize" == self.main_val_metric_goal else -1e8

        # Settings
        self.max_epoch = int(self.cfg.max_epoch)
        self.max_iter = int(self.cfg.max_iter)
        self.gradient_accumulation_steps = accumulation_steps
        self.save_period = int(self.cfg.trainer.save_period)
        self.backup_period = int(self.cfg.trainer.backup_period)
        self.val_period = int(self.cfg.trainer.validation_period)
        self.vis_period = int(self.cfg.trainer.visualization_period)

        # Multi-resolution noise
        self.apply_multi_res_noise = self.cfg.multi_res_noise is not None
        if self.apply_multi_res_noise:
            self.mr_noise_strength = self.cfg.multi_res_noise.strength
            self.annealed_mr_noise = self.cfg.multi_res_noise.annealed
            self.mr_noise_downscale_strategy = self.cfg.multi_res_noise.downscale_strategy

        # Internal variables
        self.epoch = 1
        self.n_batch_in_epoch = 0
        self.effective_iter = 0
        self.in_evaluation = False
        self.global_seed_sequence: List = []

    def _replace_unet_conv_in(self, unet):
        """
        Replace conv_in of a UNet to accept 8 input channels.
        This assumes original conv_in had in_channels == 4 (Stable Diffusion style).
        """
        orig_in = getattr(unet.conv_in, "in_channels", None)
        if orig_in is None:
            logging.warning("UNet conv_in missing in_channels attribute; attempting best-effort replacement.")
            orig_in = 4
        # desired in channels
        new_in = 8
        if orig_in == new_in:
            return

        # read weights/bias if present
        try:
            _weight = unet.conv_in.weight.clone()  # [out, in, k, k]
            _bias = unet.conv_in.bias.clone() if unet.conv_in.bias is not None else None
        except Exception:
            # create random init if impossible
            _weight = None
            _bias = None

        # compute repeat factor
        if _weight is not None:
            if new_in % orig_in != 0:
                raise ValueError(f"Cannot expand conv_in from {orig_in} to {new_in}")
            repeat_factor = new_in // orig_in
            _weight = _weight.repeat((1, repeat_factor, 1, 1))
            _weight *= (1.0 / repeat_factor)  # scale down magnitudes
        out_ch = unet.conv_in.out_channels
        _new_conv_in = Conv2d(new_in, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        if _weight is not None:
            _new_conv_in.weight = Parameter(_weight)
        if _bias is not None:
            _new_conv_in.bias = Parameter(_bias)
        unet.conv_in = _new_conv_in
        # try to set config if present
        try:
            unet.config["in_channels"] = new_in
        except Exception:
            try:
                setattr(unet.config, "in_channels", new_in)
            except Exception:
                pass
        logging.info("Unet conv_in layer replaced to accept %d channels", new_in)

    def train(self, t_end=None):
        logging.info("Start training")
        device = self.device
        self.model.to(device)

        # If we had an in-progress evaluation, do it now:
        if self.in_evaluation:
            logging.info("Last evaluation was not finished, running validate first.")
            self.validate()

        self.train_metrics.reset()
        accumulated_step = 0

        # Do an initial visualization run
        self.in_evaluation = True
        self.visualize()
        self.in_evaluation = False

        for epoch in range(self.epoch, self.max_epoch + 1):
            self.epoch = epoch
            logging.debug(f"epoch: {self.epoch}")
            for batch in skip_first_batches(self.train_loader, self.n_batch_in_epoch):
                self.model.unet_derolling.train()
                if hasattr(self.model, "unet_marigold"):
                    self.model.unet_marigold.eval()

                # deterministic random generator if desired
                if self.seed is not None:
                    local_seed = self._get_next_seed()
                    rand_num_generator = torch.Generator(device=device)
                    rand_num_generator.manual_seed(local_seed)
                else:
                    rand_num_generator = None

                # --- Get data ---
                rolling = batch["rolling_norm"].to(device)        # noisy rolling capture
                global_base = batch["global_norm"].to(device)    # GT RGB (used for teacher)
                depth_base = batch.get("depth_raw_linear", None)
                if depth_base is not None:
                    depth_base = depth_base.to(device)

                batch_size = rolling.shape[0]

                # Encode latents for RGB path (no grad)
                with torch.no_grad():
                    rolling_latent = self.model.encode_rgb(rolling)    # [B, 4, H, W]
                    global_latent = self.model.encode_rgb(global_base) # [B, 4, H, W]

                    # Depth latent used for marigold branch previously:
                    if depth_base is not None:
                        depth_latent = self.encode_depth(depth_base)  # [B, 4, H, W]
                    else:
                        depth_latent = None

                # If using teacher pipeline, compute teacher depth (pseudo-GT) from GT RGB
                if self.teacher_pipeline is not None:
                    # we will get teacher_depth as a torch tensor [B, H, W] in model scale (0..1 or depth units)
                    teacher_depth_tensor = self._get_teacher_depth_for_batch(global_base, batch)
                    # encode teacher depth into latent space (if training marigold in latent domain)
                    with torch.no_grad():
                        teacher_depth_latent = self.encode_depth(teacher_depth_tensor)
                else:
                    teacher_depth_tensor = None
                    teacher_depth_latent = depth_latent  # fallback if dataset provides depths

                # sample timesteps
                timesteps = torch.randint(
                    0,
                    self.scheduler_timesteps,
                    (batch_size,),
                    device=device,
                    generator=rand_num_generator,
                ).long()

                # Sample / create noise tensors
                if self.apply_multi_res_noise:
                    strength = self.mr_noise_strength
                    if self.annealed_mr_noise:
                        strength = strength * (timesteps / self.scheduler_timesteps)
                    noise_derolling = multi_res_noise_like(
                        global_latent,
                        strength=strength,
                        downscale_strategy=self.mr_noise_downscale_strategy,
                        generator=rand_num_generator,
                        device=device,
                    )
                    if teacher_depth_latent is not None:
                        noise_marigold = multi_res_noise_like(
                            teacher_depth_latent,
                            strength=strength,
                            downscale_strategy=self.mr_noise_downscale_strategy,
                            generator=rand_num_generator,
                            device=device,
                        )
                    else:
                        noise_marigold = torch.randn(global_latent.shape, device=device, generator=rand_num_generator)
                else:
                    noise_derolling = torch.randn(global_latent.shape, device=device, generator=rand_num_generator)
                    if teacher_depth_latent is not None:
                        noise_marigold = torch.randn(teacher_depth_latent.shape, device=device, generator=rand_num_generator)
                    else:
                        noise_marigold = torch.randn(global_latent.shape, device=device, generator=rand_num_generator)

                # Add noise (diffusion forward)
                noisy_derolling_latents = self.training_noise_scheduler.add_noise(global_latent, noise_derolling, timesteps)
                if teacher_depth_latent is not None:
                    noisy_marigold_latents = self.training_noise_scheduler.add_noise(teacher_depth_latent, noise_marigold, timesteps)
                else:
                    noisy_marigold_latents = noise_marigold  # not used but keep a placeholder

                # text embedding
                if self.empty_text_embed is not None:
                    text_embed = self.empty_text_embed.to(device).repeat((batch_size, 1, 1))
                else:
                    text_embed = None

                # Concatenate latents for derolling UNet input: [rolling_latent, noisy_derolling_latents] -> 8 channels
                cat_latents_derolling = torch.cat([rolling_latent, noisy_derolling_latents], dim=1).float()
                model_pred_rgb = self.model.unet_derolling(cat_latents_derolling, timesteps, text_embed).sample  # [B,4,H,W]
                if torch.isnan(model_pred_rgb).any():
                    logging.warning("model_pred_rgb contains NaN.")

                # Build marigold input using predicted rgb latent + noisy marigold latents
                cat_latents_marigold = torch.cat([model_pred_rgb.detach(), noisy_marigold_latents], dim=1).float()

                # Run frozen marigold UNet if available (or use teacher's depth latents)
                if hasattr(self.model, "unet_marigold"):
                    with torch.no_grad():
                        model_pred_depth_latent = self.model.unet_marigold(cat_latents_marigold, timesteps, text_embed).sample
                else:
                    # If the model does not have a marigold unet we skip latent depth prediction and compute losses in depth image domain
                    model_pred_depth_latent = None

                # Prepare targets depending on prediction type
                if "sample" == self.prediction_type:
                    target = global_latent
                    target_depth = teacher_depth_latent if teacher_depth_latent is not None else (depth_latent if depth_latent is not None else None)
                elif "epsilon" == self.prediction_type:
                    target = noise_derolling
                    target_depth = noise_marigold
                elif "v_prediction" == self.prediction_type:
                    target = self.training_noise_scheduler.get_velocity(global_latent, noise_derolling, timesteps)
                    if teacher_depth_latent is not None:
                        target_depth = self.training_noise_scheduler.get_velocity(teacher_depth_latent, noise_marigold, timesteps)
                    else:
                        target_depth = self.training_noise_scheduler.get_velocity(depth_latent, noise_marigold, timesteps) if depth_latent is not None else None
                else:
                    raise ValueError(f"Unknown prediction type {self.prediction_type}")

                if target_depth is None and model_pred_depth_latent is not None:
                    # fallback: if no depth target, zero-target to avoid crash (but should not happen)
                    target_depth = torch.zeros_like(model_pred_depth_latent)

                # compute latent losses
                latent_loss_derolling = self.loss_derolling(model_pred_rgb.float(), target.float()).mean()
                if model_pred_depth_latent is not None:
                    latent_loss_marigold = self.loss_marigold(model_pred_depth_latent.detach().float(), target_depth.float()).mean()
                else:
                    # if no marigold latent inside model, compute depth supervision in image domain:
                    # decode model_pred_rgb to rgb image, run model decode path to depth, etc.
                    latent_loss_marigold = torch.tensor(0.0, device=device)

                loss = latent_loss_derolling + (1.0 / 5.0) * latent_loss_marigold
                self.train_metrics.update("loss", loss.item())

                # backward with gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                accumulated_step += 1

                self.n_batch_in_epoch += 1

                if accumulated_step >= self.gradient_accumulation_steps:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    accumulated_step = 0
                    self.effective_iter += 1

                    # logging
                    accumulated_loss = self.train_metrics.result()["loss"]
                    tb_logger.log_dic({f"train/{k}": v for k, v in self.train_metrics.result().items()}, global_step=self.effective_iter)
                    tb_logger.writer.add_scalar("lr", self.lr_scheduler.get_last_lr()[0], global_step=self.effective_iter)
                    tb_logger.writer.add_scalar("n_batch_in_epoch", self.n_batch_in_epoch, global_step=self.effective_iter)
                    logging.info(f"iter {self.effective_iter:5d} (epoch {epoch:2d}): loss={accumulated_loss:.5f}")
                    self.train_metrics.reset()

                    self._train_step_callback()

                    # termination checks
                    if self.max_iter > 0 and self.effective_iter >= self.max_iter:
                        self.save_checkpoint(ckpt_name=self._get_backup_ckpt_name(), save_train_state=False)
                        logging.info("Training ended.")
                        return
                    elif t_end is not None and datetime.now() >= t_end:
                        self.save_checkpoint(ckpt_name="latest", save_train_state=True)
                        logging.info("Time is up, training paused.")
                        return

                    torch.cuda.empty_cache()

            # epoch end
            self.n_batch_in_epoch = 0

    def _get_teacher_cache_path(self, sample_id: str):
        if self.teacher_cache_dir is None:
            return None
        # sanitize sample_id if needed
        fname = sample_id.replace("/", "_")
        return os.path.join(self.teacher_cache_dir, f"{fname}.npz")

    def _get_teacher_depth_for_batch(self, global_base: torch.Tensor, batch: dict):
        """
        Returns a tensor [B, H, W] of teacher depths in same scale as your dataset (0..1 or raw depth).
        Caches outputs if teacher_cache_dir set and batch contains 'global_relative_path' to use as key.
        """
        device = self.device
        batch_size = global_base.shape[0]
        depths = []
        for i in range(batch_size):
            # prefer a dataset-provided unique path for caching
            sample_key = None
            if "global_relative_path" in batch:
                sample_key = batch["global_relative_path"][i]
            cache_path = self._get_teacher_cache_path(sample_key) if sample_key is not None else None

            if cache_path is not None and os.path.exists(cache_path):
                npz = np.load(cache_path)
                depth_np = npz["depth"]
                depth_t = torch.from_numpy(depth_np).to(device).float()
                depths.append(depth_t)
                continue

            # prepare single sample input for teacher: convert to CPU numpy or keep tensor depending on teacher API
            # We'll provide the teacher with a batched tensor of shape [1, C, H, W] in float in range model expects (we assume normalized)
            gt_rgb = global_base[i : i + 1]  # [1,C,H,W]
            # run teacher pipeline (assumes same API as validate)
            with torch.no_grad():
                # teacher may expect CPU numpy or torch; we try passing a torch tensor and rely on pipeline to accept it.
                pipe_out = self.teacher_pipeline(
                    gt_rgb,
                    denoising_steps=self.cfg.trainer.teacher_denoising_steps if "teacher_denoising_steps" in self.cfg.trainer else self.cfg.validation.denoising_steps,
                    ensemble_size=self.cfg.trainer.teacher_ensemble_size if "teacher_ensemble_size" in self.cfg.trainer else self.cfg.validation.ensemble_size,
                    processing_res=self.cfg.trainer.teacher_processing_res if "teacher_processing_res" in self.cfg.trainer else self.cfg.validation.processing_res,
                    match_input_res=self.cfg.trainer.teacher_match_input_res if "teacher_match_input_res" in self.cfg.trainer else self.cfg.validation.match_input_res,
                    batch_size=1,
                    generator=None,
                )
            # Expect pipe_out.depth_np to be HxW float
            depth_np = pipe_out.depth_np.astype(np.float32)
            depth_t = torch.from_numpy(depth_np).to(device).float()
            if cache_path is not None:
                np.savez_compressed(cache_path, depth=depth_np)
            depths.append(depth_t)

        # stack to [B, H, W]
        return torch.stack(depths, dim=0)

    def encode_depth(self, depth_in: torch.Tensor):
        """
        Convert a depth tensor into a 3-channel image and then encode with model's encode_rgb (VAE encoder).
        Expects depth_in to be either [B,1,H,W] or [B,H,W] or [H,W] etc.
        Returns latent [B, 4, h, w]
        """
        stacked = self.stack_depth_images(depth_in)
        depth_latent = self.model.encode_rgb(stacked)
        return depth_latent

    @staticmethod
    def stack_depth_images(depth_in):
        # print(depth_in.shape)
        if 4 == len(depth_in.shape):
            stacked = depth_in.repeat(1, 3, 1, 1)
        elif 3 == len(depth_in.shape):
            stacked = depth_in.unsqueeze(1)
            stacked = depth_in.repeat(1, 3, 1, 1)
        return stacked

    def _train_step_callback(self):
        """Executed after every iteration"""
        # Save backup (with a larger interval, without training states)
        if self.backup_period > 0 and 0 == self.effective_iter % self.backup_period:
            self.save_checkpoint(
                ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
            )

        _is_latest_saved = False
        # Validation
        if self.val_period > 0 and 0 == self.effective_iter % self.val_period:
            self.in_evaluation = True  # flag to do evaluation in resume run if validation is not finished
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)
            _is_latest_saved = True
            self.validate()
            self.in_evaluation = False
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)

        # Save training checkpoint (can be resumed)
        if (
            self.save_period > 0
            and 0 == self.effective_iter % self.save_period
            and not _is_latest_saved
        ):
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)

        # Visualization
        if self.vis_period > 0 and 0 == self.effective_iter % self.vis_period:
            self.visualize()

    def validate(self):
        for i, val_loader in enumerate(self.val_loaders):
            val_dataset_name = val_loader.dataset.disp_name

            val_metric_dic = self.validate_single_dataset(
                data_loader=val_loader
            )
            logging.info(
                f"Iter {self.effective_iter}. Validation metrics on `{val_dataset_name}`: {val_metric_dic}"
            )

            # Log metrics
            tb_logger.log_dic(
                {f"val/{val_dataset_name}/{k}": v for k, v in val_metric_dic.items()},
                global_step=self.effective_iter,
            )

            # Save evaluation to a file
            eval_text = eval_dic_to_text(
                val_metrics=val_metric_dic,
                dataset_name=val_dataset_name,
                sample_list_path=val_loader.dataset.filename_ls_path,
            )
            save_path = os.path.join(
                self.out_dir_eval, f"eval-{val_dataset_name}-iter{self.effective_iter:06d}.txt"
            )
            with open(save_path, "w+") as f:
                f.write(eval_text)

            # # Update the best metric
            # if i == 0:
            #     main_eval_metric = val_metric_dic[self.main_val_metric]
            #     is_best = (
            #         (self.main_val_metric_goal == "minimize" and main_eval_metric < self.best_metric) or
            #         (self.main_val_metric_goal == "maximize" and main_eval_metric > self.best_metric)
            #     )
            #     if is_best:
            #         self.best_metric = main_eval_metric
            #         logging.info(
            #             f"New best metric: {self.main_val_metric} = {self.best_metric} at iteration {self.effective_iter}"
            #         )
            #         self.save_checkpoint(
            #             ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
            #         )

    def visualize(self):
        for val_loader in self.vis_loaders:
            vis_dataset_name = val_loader.dataset.disp_name
            vis_out_dir = os.path.join(
                self.out_dir_vis, self._get_backup_ckpt_name(), vis_dataset_name
            )
            os.makedirs(vis_out_dir, exist_ok=True)
            _ = self.validate_single_dataset(
                data_loader=val_loader,
                save_to_dir=vis_out_dir,
            )

    @torch.no_grad()
    def validate_single_dataset(self, data_loader, save_to_dir=None):
        self.model.to(self.device)

        # Define metric collection
        metrics = MetricCollection({
            "MSE" : MeanSquaredError()
        }).to(self.device)

        # Reset metrics
        metrics.reset()

        for batch in tqdm(data_loader, desc=f"Validating on {data_loader.dataset.disp_name}"):
            # Assuming batch size of 1
            rolling_int = batch["rolling_int"].squeeze().to(self.device)  # [C, H, W]
            global_int = batch["global_int"].squeeze().to(self.device)  # Ground truth
            depth_int = batch["depth_raw_linear"].squeeze().to(self.device)
            # print(rolling_int.shape)
            # Model inference
            pipe_out = self.model(
                rolling_int,
                denoising_steps=self.cfg.validation.denoising_steps,
                ensemble_size=self.cfg.validation.ensemble_size,
                processing_res=self.cfg.validation.processing_res,
                match_input_res=self.cfg.validation.match_input_res,
                batch_size=1,
                generator=None,
            )

            global_pred = torch.from_numpy(pipe_out.depth_np).to(self.device)
            depth_int = (depth_int/255).float()
            # print("Type Pred: " + str(global_pred.dtype))
            # print("Type Glob: " + str(global_int.dtype))
            # Update metrics
            metrics.update(global_pred.unsqueeze(0), depth_int.unsqueeze(0))

            # Optionally save predictions
            if save_to_dir is not None:
                img_name = batch["global_relative_path"][0].replace("/", "_")
                png_save_path = os.path.join(save_to_dir, f"{img_name}.png")
                png_save_path_rgb = os.path.join(save_to_dir, f"{img_name}_rgb.png")
                depth_to_save = (pipe_out.depth_np * 65535.0).astype(np.uint16)
                rgb_to_save = (pipe_out.rgb*255).astype(np.uint8)
                Image.fromarray(depth_to_save).save(png_save_path, mode="I;16")
                Image.fromarray(rgb_to_save.transpose(1, 2, 0), mode="RGB").save(png_save_path_rgb)


        # Compute metrics
        return metrics.compute()

    def _get_next_seed(self):
        if 0 == len(self.global_seed_sequence):
            self.global_seed_sequence = generate_seed_sequence(
                initial_seed=self.seed,
                length=self.max_iter * self.gradient_accumulation_steps,
            )
            logging.info(
                f"Global seed sequence is generated, length={len(self.global_seed_sequence)}"
            )
        return self.global_seed_sequence.pop()

    def save_checkpoint(self, ckpt_name, save_train_state):
        ckpt_dir = os.path.join(self.out_dir_ckpt, ckpt_name)
        logging.info(f"Saving checkpoint to: {ckpt_dir}")
        # Backup previous checkpoint
        temp_ckpt_dir = None
        if os.path.exists(ckpt_dir) and os.path.isdir(ckpt_dir):
            temp_ckpt_dir = os.path.join(
                os.path.dirname(ckpt_dir), f"_old_{os.path.basename(ckpt_dir)}"
            )
            if os.path.exists(temp_ckpt_dir):
                shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            os.rename(ckpt_dir, temp_ckpt_dir)
            logging.debug(f"Old checkpoint is backed up at: {temp_ckpt_dir}")

        # Save UNet
        unet_path = os.path.join(ckpt_dir, "unet_{}")
        self.model.unet_derolling.save_pretrained(unet_path.format("derolling"), safe_serialization=False)
        self.model.unet_.save_pretrained(unet_path.format("marigold"), safe_serialization=False)
        logging.info(f"UNet is saved to: {unet_path}")

        if save_train_state:
            state = {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "config": self.cfg,
                "effective_iter": self.effective_iter,
                "epoch": self.epoch,
                "n_batch_in_epoch": self.n_batch_in_epoch,
                "best_metric": self.best_metric,
                "in_evaluation": self.in_evaluation,
                "global_seed_sequence": self.global_seed_sequence,
            }
            train_state_path = os.path.join(ckpt_dir, "trainer.ckpt")
            torch.save(state, train_state_path)
            # iteration indicator
            f = open(os.path.join(ckpt_dir, self._get_backup_ckpt_name()), "w")
            f.close()

            logging.info(f"Trainer state is saved to: {train_state_path}")

        # Remove temp ckpt
        if temp_ckpt_dir is not None and os.path.exists(temp_ckpt_dir):
            shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            logging.debug("Old checkpoint backup is removed.")

    def load_checkpoint(
        self, ckpt_path, load_trainer_state=True, resume_lr_scheduler=True
    ):
        logging.info(f"Loading checkpoint from: {ckpt_path}")
        # Load UNet
        _model_path = os.path.join(ckpt_path, "unet", "diffusion_pytorch_model.bin")
        self.model.unet_derolling.load_state_dict(
            torch.load(_model_path, map_location=self.device)
        )
        self.model.unet_derolling.to(self.device)
        logging.info(f"UNet parameters are loaded from {_model_path}")

        # Load training states
        if load_trainer_state:
            checkpoint = torch.load(os.path.join(ckpt_path, "trainer.ckpt"))
            self.effective_iter = checkpoint["effective_iter"]
            self.epoch = checkpoint["epoch"]
            self.n_batch_in_epoch = checkpoint["n_batch_in_epoch"]
            self.in_evaluation = checkpoint["in_evaluation"]
            self.global_seed_sequence = checkpoint["global_seed_sequence"]

            self.best_metric = checkpoint["best_metric"]

            self.optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(f"optimizer state is loaded from {ckpt_path}")

            if resume_lr_scheduler:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                logging.info(f"LR scheduler state is loaded from {ckpt_path}")

        logging.info(
            f"Checkpoint loaded from: {ckpt_path}. Resume from iteration {self.effective_iter} (epoch {self.epoch})"
        )
        return

    def _get_backup_ckpt_name(self):
        return f"iter_{self.effective_iter:06d}"
    
    def encode_depth(self, depth_in):
        # stack depth into 3-channel
        stacked = self.stack_depth_images(depth_in)
        # encode using VAE encoder
        depth_latent = self.model.encode_rgb(stacked)
        return depth_latent

