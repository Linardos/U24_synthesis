import monai
from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

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
        )
        # --- diffusion utilities --------------------------------------------
        self.scheduler = DDPMScheduler(num_train_timesteps=T)
        self.inferer   = DiffusionInferer(self.scheduler)
        self.lr        = lr
        self.save_hyperparameters()

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
