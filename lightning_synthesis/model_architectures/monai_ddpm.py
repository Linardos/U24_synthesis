import monai
from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from tqdm import tqdm

class MonaiDDPM(pl.LightningModule):
    def __init__(self, lr, T=1000):
        super().__init__()
        # --- noise‑predictor -------------------------------------------------
        self.unet = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=(128, 256, 256),
            attention_levels=(False, True, True),
            num_res_blocks=1,
            num_head_channels=256,
            use_flash_attention=True,
            with_conditioning=True,
            cross_attention_dim=1,
        )

        # --- diffusion utilities --------------------------------------------
        self.scheduler = DDPMScheduler(num_train_timesteps=T)
        self.inferer   = DiffusionInferer(self.scheduler)
        self.lr        = lr
        self.save_hyperparameters()
        print("Initialized conditional model.")

    # Lightning uses whatever you return from forward in predict_step / sampling
    def forward(self, x, t):                       # mimic your old API
        return self.unet(x, t)

    # ---------- training loop -----------------------------------------------
    def training_step(self, batch, _):
        images, labels = batch 
        noise      = torch.randn_like(images)
        timesteps  = torch.randint(
            0, self.scheduler.num_train_timesteps,
            (images.shape[0],), device=self.device
        ).long()

        # This call internally adds noise (q sample) & runs the UNet
        noise_pred = self.inferer(
            inputs=images,
            diffusion_model=self.unet,
            noise=noise,
            timesteps=timesteps,
            condition=labels
        )

        loss = F.mse_loss(noise_pred.float(), noise.float())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    # ---------- sampling helpers ----------------------------------------
    @torch.no_grad()
    def sample_cfg(self, label=0, N=16, size=64, guidance_scale=4.0):
        """
        Class-conditional sampling with classifier-free guidance, MONAI-style batch concat.
        """
        device = self.device
        noise  = torch.randn(N, 1, size, size, device=device)

        # timesteps
        self.scheduler.set_timesteps(self.hparams.T)
        timesteps = self.scheduler.timesteps       
        # build conditioning tensor (B, 1, cross_dim)
        if isinstance(label, int):
            label = torch.full((N,), label, device=device, dtype=torch.long)
        cond  = label.float().view(N, 1, 1)          # scalar to (B,1,1)
        uncond = -1 * torch.ones_like(cond)
        context = torch.cat([uncond, cond], dim=0)      # (2B,1,C)

        x = noise
        for t in tqdm(timesteps):
            t_b  = torch.full((N,), t, device=device, dtype=torch.long)
            t_bb = torch.cat([t_b, t_b], dim=0)         # (2B,)

            # 1 forward pass with doubled batch
            model_out = self.unet(torch.cat([x, x], dim=0), timesteps=t_bb, context=context)
            eps_uncond, eps_cond = model_out.chunk(2)

            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            x, _ = self.scheduler.step(eps, t, x)

        return x.clamp(0, 1)     # images in [0,1]



class MonaiDDPM_unconditional(pl.LightningModule):
    def __init__(self, lr, T=1000):
        super().__init__()
        # --- noise‑predictor -------------------------------------------------
        self.unet = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=(128, 256, 512, 512),   # four levels instead of three
            attention_levels=(False, True, True, True),
            num_res_blocks=2,                    # >1 res-block per level helps
            num_head_channels=64,                # spread attention across more heads
            use_flash_attention=True,
        )
        # --- diffusion utilities --------------------------------------------
        self.scheduler = DDPMScheduler(num_train_timesteps=T)
        self.inferer   = DiffusionInferer(self.scheduler)
        self.lr        = lr
        self.save_hyperparameters()
        print("Initialized unconditional model.")

    # Lightning uses whatever you return from forward in predict_step / sampling
    def forward(self, x, t):                       # mimic your old API
        return self.unet(x, t)

    # ---------- training loop -----------------------------------------------
    def training_step(self, batch, _):
        images, _ = batch 
        noise      = torch.randn_like(images)
        timesteps  = torch.randint(
            0, self.scheduler.num_train_timesteps,
            (images.shape[0],), device=self.device
        ).long()

        # This call internally adds noise (q sample) & runs the UNet
        noise_pred = self.inferer(
            inputs=images,
            diffusion_model=self.unet,
            noise=noise,
            timesteps=timesteps,
        )

        loss = F.mse_loss(noise_pred.float(), noise.float())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    # ---------- nice sampling helper ----------------------------------------
    @torch.no_grad()
    def sample(self, N=16, size=64):
        noise=torch.randn(N, 1, size, size, device=self.device)
        image, intermediates = self.inferer.sample(
            input_noise=noise,
            diffusion_model=self.unet,
            save_intermediates=True,     # optional
            intermediate_steps=100,      # optional
        )
        return image, intermediates                  # (N,1,size,size)