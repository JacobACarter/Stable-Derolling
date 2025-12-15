# An official reimplemented version of Marigold training script.
# Last modified: 2024-04-29
#
# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# If you use or adapt this code, please attribute to https://github.com/prs-eth/marigold.
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------


import logging
import os
import shutil
from datetime import datetime
from typing import List, Union
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, MeanSquaredError
from torchmetrics.collections import MetricCollection

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
from src.util.alignment import align_rgb_least_square
from src.util.seeding import generate_seed_sequence
import pickle


class MarigoldDualUnetTrainer:
    def __init__(
        self,
        cfg: OmegaConf,
        model: MarigoldRGBDepthPipeline,
        train_dataloader: DataLoader,
        device,
        base_ckpt_dir,
        out_dir_ckpt,
        out_dir_eval,
        out_dir_vis,
        accumulation_steps: int,
        val_dataloaders: List[DataLoader] = None,
        vis_dataloaders: List[DataLoader] = None,
    ):
        self.cfg: OmegaConf = cfg
        self.model: MarigoldRGBDepthPipeline = model
        self.device = device
        self.seed: Union[int, None] = (
            self.cfg.trainer.init_seed
        )  # used to generate seed sequence, set to `None` to train w/o seeding
        self.out_dir_ckpt = out_dir_ckpt
        self.out_dir_eval = out_dir_eval
        self.out_dir_vis = out_dir_vis
        self.train_loader: DataLoader = train_dataloader
        self.val_loaders: List[DataLoader] = val_dataloaders
        self.vis_loaders: List[DataLoader] = vis_dataloaders
        self.accumulation_steps: int = accumulation_steps

        # Adapt input layers
        if 8 != self.model.unet_derolling.config["in_channels"]:
            self._replace_unet_conv_in(self.model.unet_derolling)

                # Adapt input layers
        if 8 != self.model.unet_marigold.config["in_channels"]:
            self._replace_unet_conv_in(self.model.unet_marigold)

        
        #From pretrained!!!
        # model_path = "output/train_X4K_center/checkpoint/latest/unet/diffusion_pytorch_model.bin"
        # print(f"Loading Model from: {model_path}")
        # self.model.unet.load_state_dict(
        #     torch.load(model_path, map_location=device)
        # )
        # Encode empty text prompt
        self.model.encode_empty_text()
        self.empty_text_embed = self.model.empty_text_embed.detach().clone().to(device)

        self.model.unet_derolling.enable_xformers_memory_efficient_attention()
        self.model.unet_marigold.enable_xformers_memory_efficient_attention()
        for param in self.model.unet_marigold.parameters():
            param.requires_grad = False
        # Trainability
        self.model.vae.requires_grad_(False)
        self.model.text_encoder.requires_grad_(False)
        self.model.unet_derolling.requires_grad_(True)


        # Optimizer !should be defined after input layer is adapted
        lr = self.cfg.lr
        self.optimizer_derolling = Adam(self.model.unet_derolling.parameters(), lr=lr)
        self.optimizer_marigold = Adam(self.model.unet_marigold.parameters(), lr=1.5e-05)

        # LR scheduler
        lr_func = IterExponential(
            total_iter_length=self.cfg.lr_scheduler.kwargs.total_iter,
            final_ratio=self.cfg.lr_scheduler.kwargs.final_ratio,
            warmup_steps=self.cfg.lr_scheduler.kwargs.warmup_steps,
        )
        self.lr_scheduler_derolling = LambdaLR(optimizer=self.optimizer_derolling, lr_lambda=lr_func)
        self.lr_scheduler_marigold = LambdaLR(optimizer=self.optimizer_marigold, lr_lambda=lr_func)

        # Loss
        self.loss_derolling = get_loss(loss_name=self.cfg.loss_derolling.name, **(self.cfg.loss_derolling.kwargs or {}))


        self.loss_marigold = get_loss(loss_name=self.cfg.loss_marigold.name, **(self.cfg.loss_marigold.kwargs or {}))
                                       
        # Training noise scheduler
        self.training_noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(
            os.path.join(
                base_ckpt_dir,
                cfg.trainer.training_noise_scheduler.pretrained_path,
                "scheduler",
            )
        )
        self.prediction_type = self.training_noise_scheduler.config.prediction_type
        assert (
            self.prediction_type == self.model.scheduler.config.prediction_type
        ), "Different prediction types"
        self.scheduler_timesteps = (
            self.training_noise_scheduler.config.num_train_timesteps
        )

        # Eval metrics
        self.metric_funcs = [getattr(metric, _met) for _met in cfg.eval.eval_metrics]
        self.train_metrics = MetricTracker(*["loss"])
        self.val_metrics = MetricTracker(*[m.__name__ for m in self.metric_funcs])
        # main metric for best checkpoint saving
        self.main_val_metric = cfg.validation.main_val_metric
        self.main_val_metric_goal = cfg.validation.main_val_metric_goal
        assert (
            self.main_val_metric in cfg.eval.eval_metrics
        ), f"Main eval metric `{self.main_val_metric}` not found in evaluation metrics."
        self.best_metric = 1e8 if "minimize" == self.main_val_metric_goal else -1e8

        # Settings
        self.max_epoch = self.cfg.max_epoch
        self.max_iter = self.cfg.max_iter
        self.gradient_accumulation_steps = accumulation_steps
        self.save_period = self.cfg.trainer.save_period
        self.backup_period = self.cfg.trainer.backup_period
        self.val_period = self.cfg.trainer.validation_period
        self.vis_period = self.cfg.trainer.visualization_period

        # Multi-resolution noise
        self.apply_multi_res_noise = self.cfg.multi_res_noise is not None
        if self.apply_multi_res_noise:
            self.mr_noise_strength = self.cfg.multi_res_noise.strength
            self.annealed_mr_noise = self.cfg.multi_res_noise.annealed
            self.mr_noise_downscale_strategy = (
                self.cfg.multi_res_noise.downscale_strategy
            )

        # Internal variables
        self.epoch = 1
        self.n_batch_in_epoch = 0  # batch index in the epoch, used when resume training
        self.effective_iter = 0  # how many times optimizer.step() is called
        self.in_evaluation = False
        self.global_seed_sequence: List = []  # consistent global seed sequence, used to seed random generator, to ensure consistency when resuming

    def _replace_unet_conv_in(self, unet):
        # replace the first layer to accept 8 in_channels
        _weight = unet.conv_in.weight.clone()  # [320, 4, 3, 3]
        _bias = unet.conv_in.bias.clone()  # [320]
        _weight = _weight.repeat((1, 2, 1, 1))  # Keep selected channel(s)
        # half the activation magnitude
        _weight *= 0.5
        # new conv_in channel
        _n_convin_out_channel = unet.conv_in.out_channels
        _new_conv_in = Conv2d(
            8, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        _new_conv_in.weight = Parameter(_weight)
        _new_conv_in.bias = Parameter(_bias)
        
        unet.conv_in = _new_conv_in
        logging.info("Unet conv_in layer is replaced")
        # replace config
        
        unet.config["in_channels"] = 8
        logging.info("Unet config is updated")
        return

    def train(self, t_end=None):
        logging.info("Start training")

        device = self.device
        self.model.to(device)

        if self.in_evaluation:
            logging.info(
                "Last evaluation was not finished, will do evaluation before continue training."
            )
            self.validate()

        self.train_metrics.reset()
        accumulated_step = 0
        #do evaluation at beginning
        self.in_evaluation = True  # flag to do evaluation in resume run if validation is not finished
        _is_latest_saved = True
        # self.validate()
        self.visualize()
        self.in_evaluation = False
        for epoch in range(self.epoch, self.max_epoch + 1):
            self.epoch = epoch
            logging.debug(f"epoch: {self.epoch}")

            # Skip previous batches when resume
            for batch in skip_first_batches(self.train_loader, self.n_batch_in_epoch):
                self.model.unet_derolling.train()
                self.model.unet_marigold.train()
                # self.model.unet_marigold.train()

                # globally consistent random generators
                if self.seed is not None:
                    local_seed = self._get_next_seed()
                    rand_num_generator = torch.Generator(device=device)
                    rand_num_generator.manual_seed(local_seed)
                else:
                    rand_num_generator = None

                # >>> With gradient accumulation >>>

                # Get data
                rolling = batch["rolling_norm"].to(device)
                global_base = batch["global_norm"].to(device)
                depth_base = batch["depth_raw_linear"].to(device)

                # if self.gt_mask_type is not None:
                #     valid_mask_for_latent = batch[self.gt_mask_type].to(device)
                #     invalid_mask = ~valid_mask_for_latent
                #     valid_mask_down = ~torch.max_pool2d(
                #         invalid_mask.float(), 8, 8
                #     ).bool()
                #     valid_mask_down = valid_mask_down.repeat((1, 4, 1, 1))
                # else:
                #     raise NotImplementedError

                batch_size = rolling.shape[0]
                with torch.no_grad():
                    # Encode image
                    rolling_latent = self.model.encode_rgb(rolling)  # [B, 4, h, w]
                    # Encode GT depth
                    global_latent = self.model.encode_rgb(
                        global_base
                    )  # [B, 4, h, w]
                    depth_latent = self.encode_depth(
                        depth_base
                    )

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    self.scheduler_timesteps,
                    (batch_size,),
                    device=device,
                    generator=rand_num_generator,
                ).long()  # [B]

                # Sample noise
                if self.apply_multi_res_noise:
                    strength = self.mr_noise_strength
                    if self.annealed_mr_noise:
                        # calculate strength depending on t
                        strength = strength * (timesteps / self.scheduler_timesteps)
                    noise_derolling = multi_res_noise_like(
                        global_latent,
                        strength=strength,
                        downscale_strategy=self.mr_noise_downscale_strategy,
                        generator=rand_num_generator,
                        device=device,
                    )
                    # strength = self.mr_noise_strength
                    # if self.annealed_mr_noise:
                    #     # calculate strength depending on t
                    #     strength = strength * (timesteps / self.scheduler_timesteps)
                    noise_marigold = multi_res_noise_like(
                        depth_latent,
                        strength=strength,
                        downscale_strategy=self.mr_noise_downscale_strategy,
                        generator=rand_num_generator,
                        device=device,
                    )
                else:
                    noise_derolling = torch.randn(
                        global_latent.shape,
                        device=device,
                        generator=rand_num_generator,
                    )  # [B, 4, h, w]
                    noise_marigold = torch.randn(
                        depth_latent.shape,
                        device=device,
                        generator=rand_num_generator,
                    )  # [B, 4, h, w]

                # Add noise to the latents (diffusion forward process)
                noisy_derolling_latents = self.training_noise_scheduler.add_noise(
                    global_latent, noise_derolling, timesteps
                )  # [B, 4, h, w]\
                noisy_marigold_latents = self.training_noise_scheduler.add_noise(
                    depth_latent, noise_marigold, timesteps
                )

                # Text embedding
                text_embed = self.empty_text_embed.to(device).repeat(
                    (batch_size, 1, 1)
                )  # [B, 77, 1024]

                # Concat rgb and depth latents
                cat_latents_derolling = torch.cat(
                    [rolling_latent, noisy_derolling_latents], dim=1
                )  # [B, 8, h, w]
                cat_latents_derolling = cat_latents_derolling.float()

                # Predict the noise residual
                model_pred_rgb = self.model.unet_derolling(
                    cat_latents_derolling, timesteps, text_embed
                ).sample  # [B, 4, h, w]
                if torch.isnan(model_pred_rgb).any():
                    logging.warning("model_pred contains NaN.")

                cat_latents_marigold = torch.cat(
                    [model_pred_rgb, noisy_marigold_latents], dim=1
                )
                cat_latents_marigold = cat_latents_marigold.float()

                model_pred_depth = self.model.unet_marigold(
                    cat_latents_marigold, timesteps, text_embed
                ).sample

                if torch.isnan(model_pred_depth).any():
                    logging.warning("model_pred contains NaN.")

                # Get the target for loss depending on the prediction type
                # self.prediction_type = "sample"
                if "sample" == self.prediction_type:
                    target = global_latent
                elif "epsilon" == self.prediction_type:
                    target = noise_derolling
                elif "v_prediction" == self.prediction_type:
                    target = self.training_noise_scheduler.get_velocity(
                        global_latent, noise_derolling, timesteps
                    )  # [B, 4, h, w]
                    target_depth = self.training_noise_scheduler.get_velocity(
                        depth_latent, noise_marigold, timesteps
                    )  # [B, 4, h, w]
                else:
                    raise ValueError(f"Unknown prediction type {self.prediction_type}")
                
                # target_depth = depth_latent

                # Masked latent loss
                # if self.gt_mask_type is not None:
                #     latent_loss = self.loss(
                #         model_pred[valid_mask_down].float(),
                #         target[valid_mask_down].float(),
                #     )
                # else:
                # print("Pred (marigold):", model_pred_depth.min().item(), model_pred_depth.max().item())
                # print("Target (depth latent):", target_depth.min().item(), target_depth.max().item())
                latent_loss_derolling = self.loss_derolling(model_pred_rgb.float(), target.float())
                latent_loss_marigold = self.loss_marigold(model_pred_depth.float(), target_depth.float())
                # print(f"Loss Split {latent_loss_derolling}:{latent_loss_marigold}")
                loss = latent_loss_derolling.mean()+(1/5)*latent_loss_marigold.mean()
                # print(loss)
                self.train_metrics.update("loss", loss.item())

                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                accumulated_step += 1

                self.n_batch_in_epoch += 1
                # Practical batch end

                # Perform optimization step
                if accumulated_step >= self.gradient_accumulation_steps:
                    self.optimizer_derolling.step()
                    self.optimizer_marigold.step()
                    self.lr_scheduler_derolling.step()
                    self.lr_scheduler_marigold.step()
                    self.optimizer_derolling.zero_grad()
                    self.optimizer_marigold.zero_grad()
                    accumulated_step = 0

                    self.effective_iter += 1

                    # Log to tensorboard
                    accumulated_loss = self.train_metrics.result()["loss"]
                    tb_logger.log_dic(
                        {
                            f"train/{k}": v
                            for k, v in self.train_metrics.result().items()
                        },
                        global_step=self.effective_iter,
                    )
                    tb_logger.writer.add_scalar(
                        "lr",
                        self.lr_scheduler_derolling.get_last_lr()[0],
                        global_step=self.effective_iter,
                    )
                    tb_logger.writer.add_scalar(
                        "n_batch_in_epoch",
                        self.n_batch_in_epoch,
                        global_step=self.effective_iter,
                    )
                    logging.info(
                        f"iter {self.effective_iter:5d} (epoch {epoch:2d}): loss={accumulated_loss:.5f}"
                    )
                    self.train_metrics.reset()

                    # Per-step callback
                    self._train_step_callback()

                    # End of training
                    if self.max_iter > 0 and self.effective_iter >= self.max_iter:
                        self.save_checkpoint(
                            ckpt_name=self._get_backup_ckpt_name(),
                            save_train_state=False,
                        )
                        logging.info("Training ended.")
                        return
                    # Time's up
                    elif t_end is not None and datetime.now() >= t_end:
                        self.save_checkpoint(ckpt_name="latest", save_train_state=True)
                        logging.info("Time is up, training paused.")
                        return

                    torch.cuda.empty_cache()
                    # <<< Effective batch end <<<

            # Epoch end
            self.n_batch_in_epoch = 0

    def encode_depth(self, depth_in):
        # stack depth into 3-channel
        stacked = self.stack_depth_images(depth_in)
        # encode using VAE encoder
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
        unet_path = os.path.join(ckpt_dir, "unet")
        self.model.unet_derolling.save_pretrained(unet_path, safe_serialization=False)
        logging.info(f"UNet is saved to: {unet_path}")

        if save_train_state:
            state_derolling = {
                "optimizer": self.optimizer_derolling.state_dict(),
                "lr_scheduler": self.lr_scheduler_derolling.state_dict(),
                "config": self.cfg,
                "effective_iter": self.effective_iter,
                "epoch": self.epoch,
                "n_batch_in_epoch": self.n_batch_in_epoch,
                "best_metric": self.best_metric,
                "in_evaluation": self.in_evaluation,
                "global_seed_sequence": self.global_seed_sequence,
            }
            state_marigold = {
                "optimizer": self.optimizer_marigold.state_dict(),
                "lr_scheduler": self.lr_scheduler_marigold.state_dict(),
                "config": self.cfg,
                "effective_iter": self.effective_iter,
                "epoch": self.epoch,
                "n_batch_in_epoch": self.n_batch_in_epoch,
                "best_metric": self.best_metric,
                "in_evaluation": self.in_evaluation,
                "global_seed_sequence": self.global_seed_sequence,
            }
            train_state_path = os.path.join(ckpt_dir, "trainer_{}.ckpt")
            torch.save(state_derolling, train_state_path.format("derolling"))
            torch.save(state_marigold, train_state_path.format("marigold"))
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

