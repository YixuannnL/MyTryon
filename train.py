import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, LMSDiscreteScheduler

from models.Unet import UnetA128, UnetB128
from datasets.tryonData import TryonDataset

from opt import get_opts

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint 
# TQDMProgressBar
# from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning.loggers import TensorBoardLogger

from easydict import EasyDict as edict
import math
import itertools
from einops import rearrange, reduce, repeat

import pdb

import os
os.environ['CURL_CA_BUNDLE'] = ''

seed_everything(1234, workers=True)

def freeze_params(params):
    for param in params:
        param.requires_grad = False

class TRYONSystem(LightningModule):
    '''
    1. 对和时间步拼接的两张pose图片处理：先映射到latent space，再加噪，再过fully connection使其能与时间步拼接
    '''
    def __init__(self,hparams,):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        d_time_emb = self.hparams.channels * 4
        
        self.time_embed = nn.Sequential(
            nn.Linear(self.hparams.channels, d_time_emb),
            nn.SiLU(),
            nn.Linear(d_time_emb, d_time_emb),
        )
        
        self.FC1 = nn.Sequential(
            nn.Linear(int((self.hparams.size) * (self.hparams.size)), d_time_emb),
            nn.SiLU(),
            nn.Linear(d_time_emb, d_time_emb),
        )
        self.FC2 = nn.Sequential(
            nn.Linear(int((self.hparams.size) * (self.hparams.size)), d_time_emb),
            nn.SiLU(),
            nn.Linear(d_time_emb, d_time_emb),
        )
        
        self.Conv11 = nn.Conv2d(self.hparams.in_channels, 1, 1)
        
        self.vae = AutoencoderKL(in_channels=4)
        self.vae.from_pretrained(self.hparams.pretrained_model_name_or_path, subfolder="vae")
        self.UnetA = UnetA128(in_channels=self.hparams.in_channels, 
                              out_channels=self.hparams.out_channels, 
                              channels=self.hparams.channels,
                            #   attention_levels=self.hparams.attention_levels,
                              if_down=False, 
                            #   channels_multipliers=self.hparams.channels_multipliers,
                              n_heads=self.hparams.n_heads,
                              tf_layers=self.hparams.tf_layers,
                              d_cond=self.hparams.d_cond)
        self.UnetB = UnetB128(in_channels=self.hparams.in_channels, 
                              out_channels=self.hparams.out_channels, 
                              channels=self.hparams.channels,
                            #   attention_levels=self.hparams.attention_levels,
                              if_down=False, 
                            #   channels_multipliers=self.hparams.channels_multipliers,
                              n_heads=self.hparams.n_heads,
                              tf_layers=self.hparams.tf_layers,
                              d_cond=self.hparams.d_cond)
        
        self.noise_scheduler = DDPMScheduler.from_config(self.hparams.pretrained_model_name_or_path, subfolder="scheduler")
        
    def time_step_embedding(self, time_steps: torch.Tensor, max_period: int = 10000):
        """
        ## Create sinusoidal time step embeddings

        :param time_steps: are the time steps of shape `[batch_size]`
        :param max_period: controls the minimum frequency of the embeddings.
        """
        # $\frac{c}{2}$; half the channels are sin and the other half is cos,
        half = self.hparams.channels // 2
        # $\frac{1}{10000^{\frac{2i}{c}}}$
        frequencies = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=time_steps.device)
        # $\frac{t}{10000^{\frac{2i}{c}}}$
        args = time_steps[:, None].float() * frequencies[None]
        # $\cos\Bigg(\frac{t}{10000^{\frac{2i}{c}}}\Bigg)$ and $\sin\Bigg(\frac{t}{10000^{\frac{2i}{c}}}\Bigg)$
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
    def forward(self, x):
        
        freeze_params(self.vae.parameters())
        
        # person_pose_emd = nn.Conv2d()
        # pdb.set_trace()
        person_pose_emd = rearrange(self.Conv11(x.person_pose_latents), 'b 1 h w -> b h w')
        person_pose_emd = rearrange(person_pose_emd, 'b h w -> b (h w)')
        person_pose_emd = self.FC1(person_pose_emd)
        
        garment_pose_emd = rearrange(self.Conv11(x.garment_pose_latents), 'b 1 h w -> b h w')
        garment_pose_emd = rearrange(garment_pose_emd, 'b h w -> b (h w)')                                
        garment_pose_emd = self.FC2(garment_pose_emd)
        
        t_emb = self.time_step_embedding(x.timesteps)
        t_emb = self.time_embed(t_emb)
        
        cat_time_emd = torch.cat([person_pose_emd, garment_pose_emd, t_emb], 1)
        
        z_t = torch.cat([x.noisy_latents,x.agnostic_latents], 1)
        # 三个Unet可以分开训练
        # 1
        # pdb.set_trace()
        cond = self.UnetB(x.segment_latents, cat_time_emd)
        noise_pred = self.UnetA(z_t, cat_time_emd, cond)
        # 2
        # x = self.parallelUnet256(x.time_steps, person_pose_emd, garment_pose_emd, z_t, x.segment_latents, x.down_sample)
        # 3
        # x = self.SRUnet(x.timesteps,x.down_sample)
        return noise_pred
    
    def setup(self,stage=None):
        dataset = TryonDataset(self.hparams.root_dir)
        length = len(dataset)
        self.train_dataset, self.val_dataset = \
            random_split(dataset,
                         [length-self.hparams.val_size, self.hparams.val_size])
            
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            pin_memory=True
        )
        
    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=self.hparams.num_workers,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)
        
    def configure_optimizers(self):
        self.optimizer1 = torch.optim.AdamW(
            itertools.chain(self.UnetA.parameters(),self.UnetB.parameters()),  # only optimize the embeddings
            lr=self.hparams.lr,
            betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
            weight_decay=self.hparams.adam_weight_decay,
            eps=self.hparams.adam_epsilon,)
        self.optimizer2 = torch.optim.Adam(
            itertools.chain(self.time_embed.parameters(),self.FC1.parameters(),self.FC2.parameters()),
            lr = self.hparams.lr,
        )
        scheduler1 = CosineAnnealingLR(self.optimizer1,
                                       T_max=self.hparams.num_epochs,
                                       eta_min=self.hparams.lr/1e2)
        scheduler2 = CosineAnnealingLR(self.optimizer2,
                                       T_max=self.hparams.num_epochs,
                                       eta_min=self.hparams.lr/1e2)
        return [self.optimizer1,self.optimizer2], [scheduler1,scheduler2]
        
       
    def training_step(self, batch):
        
        # Convert images to latent space
        latents = self.vae.encode(batch['image']).latent_dist.sample().detach()
        latents = latents * 0.18215
        agnostic_latents = self.vae.encode(batch['agnostic']).latent_dist.sample().detach()
        agnostic_latents = agnostic_latents * 0.18215
        segmented_latents = (self.vae.encode(batch['segment']).latent_dist.sample().detach()) * 0.18215
        
        person_pose = (self.vae.encode(batch['personpose']).latent_dist.sample().detach()) * 0.18215
        
        garment_pose = self.vae.encode(batch['garmentpose']).latent_dist.sample().detach()
        garment_pose = garment_pose * 0.18215
        
        
        # down_sample_latents = self.vae.encode(batch['down_sample']).latent_dist.sample().detach()
        # down_sample_latents = down_sample_latents * 0.18125
        # person_pose_image = batch
        
        # Sample noise then add to the latents
        noise = torch.randn(latents.shape)
        batch_size = latents.shape[0]
        
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (batch_size,),
        ).long()
        
        # Add noise to the latents according to the noise magnituge at each timestep
        # Forward diffusion process
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        x = edict({
            'noisy_latents':noisy_latents,
            'agnostic_latents':agnostic_latents,
            'segment_latents':segmented_latents,
            # 'down_sample':down_sample_latents,
            'timesteps':timesteps,
            'person_pose_latents':person_pose,
            'garment_pose_latents':garment_pose,
        })
        pdb.set_trace()
        noise_pred = self(x)
        
        loss_mle = F.mse_loss(noise_pred, noise, reduction='none').mean([1, 2, 3]).mean() # mean([1, 2, 3]) 表示对每个样本的所有通道、所有像素点的值求平均，得到一个形状为 (batch_size,) 的张量，其中每个元素表示对应样本的平均值
        
        self.log('train/loss', loss_mle)
        
        return loss_mle
    
    def validation_step(self, batch, batch_idx):
        # Convert images to latent space
        # pdb.set_trace()
        latents = self.vae.encode(batch['image']).latent_dist.sample().detach()
        latents = latents * 0.18215
        agnostic_latents = self.vae.encode(batch['agnostic']).latent_dist.sample().detach()
        agnostic_latents = agnostic_latents * 0.18215
        segmented_latents = self.vae.encode(batch['segment']).latent_dist.sample().detach()
        segmented_latents = segmented_latents * 0.18215
        
        person_pose = (self.vae.encode(batch['personpose']).latent_dist.sample().detach()) * 0.18215
        
        garment_pose = self.vae.encode(batch['garmentpose']).latent_dist.sample().detach()
        garment_pose = garment_pose * 0.18215
        
        # Sample noise then add to the latents
        noise = torch.randn(latents.shape).to(latents.device)
        batch_size = latents.shape[0]
        
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (batch_size,),
        ).long().to(latents.device)
        
        # Add noise to the latents according to the noise magnituge at each timestep
        # Forward diffusion process
        # pdb.set_trace()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        x = edict({
            'noisy_latents':noisy_latents,
            'agnostic_latents':agnostic_latents,
            'segment_latents':segmented_latents,
            # 'down_sample':down_sample_latents,
            'timesteps':timesteps,
            'person_pose_latents':person_pose,
            'garment_pose_latents':garment_pose,
        })
        
        noise_pred = self(x)
        
        loss_mle = F.mse_loss(noise_pred, noise, reduction='none').mean([1, 2, 3]).mean() # mean([1, 2, 3]) 表示对每个样本的所有通道、所有像素点的值求平均，得到一个形状为 (batch_size,) 的张量，其中每个元素表示对应样本的平均值
        
        log = {'val_loss': loss_mle}
        
        return log
    
    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        
        self.log('val/loss',mean_loss,prog_bar=True)
        
if __name__ == '__main__':
    hparams = get_opts()
    # pdb.set_trace()
    tryonsystem = TRYONSystem(hparams)
    
    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.exp_name}',
                              filename='{epoch:d}',
                              monitor='val/loss',
                              mode='min',
                              save_top_k=5)
    # pbar = TQDMProgressBar(refresh_rate=1)
    # callbacks = [ckpt_cb, pbar]
    
    logger = TensorBoardLogger(save_dir='logs',
                               name=hparams.exp_name,
                               default_hp_metric=False)
    
    trainer = Trainer(max_epochs=hparams.num_epochs,
                      checkpoint_callback=ckpt_cb,
                    #   callbacks=callbacks,
                      resume_from_checkpoint=hparams.ckpt_path,
                      logger=logger,
                    #   early_stop_callback=None,
                      weights_summary=None,
                      progress_bar_refresh_rate=1,
                      enable_model_summary=True,
                      gpus=hparams.num_gpus,
                    #   distributed_backend='ddp' if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=1,
                      benchmark=True)
    trainer.fit(tryonsystem)
        
        
        
        