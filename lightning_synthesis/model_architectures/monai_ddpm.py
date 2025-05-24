import monai
from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, PNDMScheduler, DDIMScheduler   
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics.image import (
    StructuralSimilarityIndexMeasure,
    MultiScaleStructuralSimilarityIndexMeasure,
)

# ---- tiny Sobel-based HF loss --------------------------------------
class HighFreqLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        sobel = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32)
        self.register_buffer("kx", sobel[None, None] / 8)          # 1×1×3×3
        self.register_buffer("ky", sobel.t()[None, None] / 8)

    def forward(self, x, y):
        groups  = x.shape[1]             # 1 ch or 3 ch
        kx, ky  = self.kx.repeat(groups,1,1,1), self.ky.repeat(groups,1,1,1)
        gx = F.conv2d(x, kx, padding=1, groups=groups) - F.conv2d(y, kx, padding=1, groups=groups)
        gy = F.conv2d(x, ky, padding=1, groups=groups) - F.conv2d(y, ky, padding=1, groups=groups)
        return (gx.pow(2) + gy.pow(2)).mean()


class MonaiDDPM(pl.LightningModule):
    def __init__(self, lr, T=1000, log_every=500, w_mse=0.85, w_ssim=0.1):
        super().__init__()
        # --- noise‑predictor -------------------------------------------------
        self.unet = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=(64, 128, 256, 512, 512),
            attention_levels=(False, False, True, True, True),
            num_res_blocks=2,
            num_head_channels=64,
            use_flash_attention=True,
            with_conditioning=True,
            cross_attention_dim=1,
        )

        self.w_mse  = w_mse
        self.w_ssim = w_ssim
        self.w_hf   = 1.0 - (w_mse + w_ssim) 
        # --- diffusion utilities --------------------------------------------
        self.scheduler = DDPMScheduler(num_train_timesteps=T)
        # self.scheduler = PNDMScheduler(num_train_timesteps=T)
        # self.inferer   = DiffusionInferer(self.scheduler)
        self.lr        = lr
        self.ssim      = MultiScaleStructuralSimilarityIndexMeasure(
                        data_range=2.0,  # because images live in [-1,1]
                        gaussian_kernel=False
                     )
        self.hf = HighFreqLoss()     # weight chosen later

        self.log_every = log_every

        print(f"Loss function using {self.w_mse} MSE and {self.w_ssim} similarity measure {self.ssim}, and {self.w_hf} high frequency")
        
        self.save_hyperparameters()
        print("Initialized conditional model.")

    # Lightning uses whatever you return from forward in predict_step / sampling
    def forward(self, x, t):                       # mimic your old API
        return self.unet(x, t)

    # ---------- training loop -----------------------------------------------
    def training_step(self, batch, _):
        x0, labels = batch                # x0 ∈ [-1,1]
        eps       = torch.randn_like(x0)  # noise
        t         = torch.randint(
                       0, self.scheduler.num_train_timesteps,
                       (x0.size(0),), device=self.device).long() # timestep

        # add noise → xt
        xt = self.scheduler.add_noise(x0, eps, t)

        # predict epŝ
        eps_hat = self.unet(xt, t, context=labels)

        # This call internally adds noise (q sample) & runs the UNet
        # eps_hat = self.inferer(
        #     inputs=x0,
        #     diffusion_model=self.unet,
        #     noise=eps,
        #     timesteps=t,
        #     condition=labels
        # )

        # """ ---- MSE + SSIM loss-------------------
        mse = F.mse_loss(eps_hat.float(), eps.float()) # minimize cost of predicting noise

        if self.w_mse == 1.0:
            self.log("train_loss", loss, prog_bar=True)
        elif self.w_hf == 0:
            # reconstruct x0̂ for MS-SSIM
            alpha_bar = self.scheduler.alphas_cumprod[t].view(-1, 1, 1, 1)
            x0_hat = (xt - torch.sqrt(1 - alpha_bar) * eps_hat) / torch.sqrt(alpha_bar)
            x0_hat = x0_hat.clamp(-1, 1)

            ssim = 1.0 - self.ssim(                    # 1-SSIM ∈ [0,2]
                (x0_hat + 1) / 2,                     # SSIM expects [0,1]
                (x0     + 1) / 2,
            )

            loss = self.w_mse * mse + self.w_ssim * ssim

            self.log_dict({"mse": mse, "ssim_loss": ssim, "train_loss": loss},
                        prog_bar=True, logger=True)
        else:
            # reconstruct x0̂ for MS-SSIM & HF
            alpha_bar = self.scheduler.alphas_cumprod[t].view(-1,1,1,1)
            x0_hat = (xt - torch.sqrt(1 - alpha_bar) * eps_hat) / torch.sqrt(alpha_bar)
            x0_hat = x0_hat.clamp(-1, 1)

            ms_ssim = 1.0 - self.ssim((x0_hat+1)/2, (x0+1)/2)
            hf_loss = self.hf(x0_hat, x0)          # still in [-1,1] range

            loss = (
                self.w_mse * mse +
                self.w_ssim * ms_ssim +
                self.w_hf * hf_loss                 # << tiny HF weight
            )

            self.log_dict(
                {"mse": mse, "ms_ssim": ms_ssim, "hf": hf_loss, "train_loss": loss},
                prog_bar=True, logger=True)

        return loss

    # ------------------------------------------------------------------
    def validation_step(self, batch, _):
        x0, labels = batch
        eps   = torch.randn_like(x0)
        t     = torch.randint(
                0, self.scheduler.num_train_timesteps,
                (x0.size(0),), device=self.device).long()
        xt    = self.scheduler.add_noise(x0, eps, t)
        eps_hat = self.unet(xt, t, context=labels)

        mse = F.mse_loss(eps_hat.float(), eps.float())

        # reconstruct x0_hat (needed for SSIM / HF)
        alpha_bar = self.scheduler.alphas_cumprod[t].view(-1,1,1,1)
        x0_hat = (xt - torch.sqrt(1-alpha_bar)*eps_hat) / torch.sqrt(alpha_bar)
        x0_hat = x0_hat.clamp(-1, 1)

        ms_ssim = 1.0 - self.ssim((x0_hat+1)/2, (x0+1)/2)
        hf_loss = self.hf(x0_hat, x0)

        val_loss = (
            self.w_mse * mse +
            self.w_ssim * ms_ssim +
            self.w_hf  * hf_loss
        )

        self.log_dict(
            {"val_loss": val_loss,
            "val_mse": mse,
            "val_ms_ssim": ms_ssim,
            "val_hf": hf_loss},
            prog_bar=False, logger=True, on_step=False, on_epoch=True
        )
        return val_loss


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    # ---------- sampling helpers ----------------------------------------
    @torch.no_grad()
    def sample(self, label=0, N=16, size=256, guidance_scale=4.0, fp16=True):
        if fp16:
            autocast_ctx = torch.autocast("cuda", dtype=torch.float16)
        else:
            from contextlib import nullcontext
            autocast_ctx = nullcontext()

        device = self.device
        noise = torch.randn(N, 1, size, size, device=device, dtype=torch.float16 if fp16 else torch.float32)


        # timesteps
        timesteps = self.scheduler.timesteps       
        # build conditioning tensor (B, 1, cross_dim)
        if isinstance(label, int):
            label = torch.full((N,), label, device=device, dtype=torch.long)
        cond  = label.float().view(N, 1, 1)          # scalar to (B,1,1)
        uncond = -1 * torch.ones_like(cond)
        context = torch.cat([uncond, cond], dim=0).to(noise.dtype)      # (2B,1,C)

        x = noise
        
        with autocast_ctx:
            for t in tqdm(timesteps):
                t_b  = torch.full((N,), t, device=device, dtype=torch.long)
                t_bb = torch.cat([t_b, t_b], dim=0)         # (2B,)

                # 1 forward pass with doubled batch
                model_out = self.unet(torch.cat([x, x], dim=0), timesteps=t_bb, context=context)
                eps_uncond, eps_cond = model_out.chunk(2)

                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
                x, _ = self.scheduler.step(eps, t, x)

        # after sampling
        x = (x + 1) / 2 # <- THIS ASSUMES TRAINING WITH [-1,1] WHICH WE HERE SCALE TO [0,1]
        x = x.clamp(0, 1).float() # back to fp32 for NIFTI
        return x

        # return x.clamp(0, 1)     # images in [0,1]
    # ---------- sampling helpers ----------------------------------------
    # @torch.no_grad()
    # def sample_fast(self, label=0, N=16, size=64, guidance_scale=4.0, num_inference_steps=25, fp16=True): # use this to infer faster, using a different scheduler than the one in training
    #     if fp16:
    #         autocast_ctx = torch.autocast("cuda", dtype=torch.float16)
    #     else:
    #         from contextlib import nullcontext
    #         autocast_ctx = nullcontext()

    #     device = self.device
    #     noise = torch.randn(N, 1, size, size, device=device, dtype=torch.float16 if fp16 else torch.float32)


    #     # timesteps
    #     fast_scheduler = DDIMScheduler(
    #         num_train_timesteps = self.scheduler.num_train_timesteps,
    #         schedule            = getattr(self.scheduler, "schedule", "linear_beta"),
    #         **getattr(self.scheduler, "_schedule_args", {})
    #     )

    #     fast_scheduler.set_timesteps(num_inference_steps=num_inference_steps)  
    #     timesteps = fast_scheduler.timesteps       
    #     # build conditioning tensor (B, 1, cross_dim)
    #     if isinstance(label, int):
    #         label = torch.full((N,), label, device=device, dtype=torch.long)
    #     cond  = label.float().view(N, 1, 1)          # scalar to (B,1,1)
    #     uncond = -1 * torch.ones_like(cond)
    #     context = torch.cat([uncond, cond], dim=0)      # (2B,1,C)

    #     x = noise
        
    #     with autocast_ctx:
    #         for t in tqdm(timesteps):
    #             t_b  = torch.full((N,), t, device=device, dtype=torch.long)
    #             t_bb = torch.cat([t_b, t_b], dim=0)         # (2B,)

    #             # 1 forward pass with doubled batch
    #             model_out = self.unet(torch.cat([x, x], dim=0), timesteps=t_bb, context=context)
    #             eps_uncond, eps_cond = model_out.chunk(2)

    #             eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
    #             x, _ = fast_scheduler.step(eps, t, x)

    #     # after sampling
    #     x = (x + 1) / 2 # <- THIS ASSUMES TRAINING WITH [-1,1] WHICH WE HERE SCALE TO [0,1]
    #     x = x.clamp(0, 1).float() # back to fp32 for NIFTI
    #     return x

