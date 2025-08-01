import torch, torch.nn as nn, torch.nn.functional as F
import pytorch_lightning as pl

# ---------------------------------------------------------------------
#  1.   building blocks
# ---------------------------------------------------------------------
class ResBlockG(nn.Module):
    """Generator residual block with conditional BN."""
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.gn1 = nn.Linear(emb_dim, in_ch)           # γ
        self.gb1 = nn.Linear(emb_dim, in_ch)           # β
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)

        self.bn2 = nn.BatchNorm2d(out_ch)
        self.gn2 = nn.Linear(emb_dim, out_ch)
        self.gb2 = nn.Linear(emb_dim, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, y_emb):
        def mod_bn(bn, g, b, h):
            out = bn(h)
            γ, β = g(y_emb).unsqueeze(2).unsqueeze(3), b(y_emb).unsqueeze(2).unsqueeze(3)
            return γ * out + β
        h = F.relu(mod_bn(self.bn1, self.gn1, self.gb1, x))
        h = self.conv1(h)
        h = F.relu(mod_bn(self.bn2, self.gn2, self.gb2, h))
        h = self.conv2(h)
        return h + self.skip(x)

class ResBlockD(nn.Module):
    """Discriminator residual block using spectral-norm convs."""
    def __init__(self, in_ch, out_ch, downsample=True):
        super().__init__()
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 3, 1, 1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_ch, out_ch, 3, 1, 1))
        self.down = downsample
        self.skip = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))
    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = self.conv2(h)
        if self.down:
            h = F.avg_pool2d(h, 2)
        s = self.skip(x)
        if self.down:
            s = F.avg_pool2d(s, 2)
        return h + s
# ---------------------------------------------------------------------
#  2.   generator & discriminator
# ---------------------------------------------------------------------
class Generator(nn.Module):
    """
    7-stage generator producing 1-channel 512×512 images in [-1,1].
    """
    def __init__(self, z_dim, emb_dim, n_classes):
        super().__init__()
        self.embed = nn.Embedding(n_classes, emb_dim)

        ch_top = 512
        self.fc = nn.Linear(z_dim, 4 * 4 * ch_top)

        self.blocks = nn.ModuleList([
            ResBlockG(ch_top,      ch_top,    emb_dim),   # → 8×8
            ResBlockG(ch_top,      ch_top//2, emb_dim),   # → 16×16
            ResBlockG(ch_top//2,   ch_top//4, emb_dim),   # → 32×32
            ResBlockG(ch_top//4,   ch_top//8, emb_dim),   # → 64×64
            ResBlockG(ch_top//8,   ch_top//16, emb_dim),  # → 128×128
            ResBlockG(ch_top//16,  ch_top//32, emb_dim),  # → 256×256
            ResBlockG(ch_top//32,  ch_top//32, emb_dim)   # → 512×512
        ])

        # *** only change: output 1 channel instead of 3
        self.to_img = nn.Conv2d(ch_top//32, 1, kernel_size=3, padding=1)

    def forward(self, z, y):
        y_emb = self.embed(y)
        h = self.fc(z).view(z.size(0), 512, 4, 4)
        for blk in self.blocks:
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            h = blk(h, y_emb)
        return torch.tanh(self.to_img(F.relu(h)))

class Discriminator(nn.Module):
    """
    7 spectral-norm ResBlocks, projection conditional WGAN critic.
    """
    def __init__(self, n_classes, base_ch=128):
        super().__init__()
        ch = base_ch
        self.blocks = nn.ModuleList([
            # *** only change: first block in_ch=1
            ResBlockD(1,      ch,     downsample=True),   # 256×256
            ResBlockD(ch,     2*ch,   downsample=True),   # 128×128
            ResBlockD(2*ch,   4*ch,   downsample=True),   # 64×64
            ResBlockD(4*ch,   4*ch,   downsample=True),   # 32×32
            ResBlockD(4*ch,   4*ch,   downsample=True),   # 16×16
            ResBlockD(4*ch,   4*ch,   downsample=True),   # 8×8
            ResBlockD(4*ch,   4*ch,   downsample=False)   # 8×8
        ])

        self.linear = nn.utils.spectral_norm(nn.Linear(4*ch, 1))
        self.embed  = nn.utils.spectral_norm(nn.Embedding(n_classes, 4*ch))

    def forward(self, x, y):
        h = x
        for blk in self.blocks:
            h = blk(h)
        h = F.relu(h).sum(dim=(2, 3))
        out = self.linear(h) + (self.embed(y) * h).sum(dim=1, keepdim=True)
        return out.squeeze(1)

# ---------------------------------------------------------------------
#  3.   LightningModule
# ---------------------------------------------------------------------

class ConditionalWGAN(pl.LightningModule):
    """
    Projection-style conditional WGAN-GP
      • critic update:  n_critic times per G step
      • GP λ:           grad_penalty_weight
      • opt betas:      (0.0, 0.9) as recommended by Gulrajani et al.
    """
    def __init__(self, n_classes, z_dim=128, lr=1e-4, n_critic=1,
                 grad_penalty_weight=100.0):
        super().__init__()
        self.save_hyperparameters()
        self.G = Generator(z_dim, emb_dim=128, n_classes=n_classes)
        self.D = Discriminator(n_classes)          # behaves as a critic (no σ)

        self.automatic_optimization = False        # manual opt loop
        torch.autograd.set_detect_anomaly(True)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.orthogonal_(m.weight, gain=0.8)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)



    # --------------------------- utils ------------------------------
    def _gradient_penalty(self, real, fake, labels):
        with torch.autocast(device_type="cuda", enabled=False):
            eps = torch.rand_like(real)          # ← same dtype & device as real
            interp = eps * real + (1 - eps) * fake
            interp.requires_grad_(True)

            interp_logits = self.D(interp, labels)
            grad = torch.autograd.grad(
                outputs=interp_logits, inputs=interp,
                grad_outputs=torch.ones_like(interp_logits),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]

            gp = (grad.flatten(1).norm(2, dim=1) - 1).pow(2).mean()
        return gp


    # --------------------------- training ---------------------------
    def training_step(self, batch, batch_idx):
        x_real, y = batch
        opt_g, opt_d = self.optimizers()
        # ------------------------------------------------------------
        # 1. Train critic D (n_critic steps)
        # ------------------------------------------------------------
        for _ in range(self.hparams.n_critic):
            z = torch.randn(x_real.size(0), self.hparams.z_dim, device=self.device)
            x_fake = self.G(z, y).detach()

            d_real = self.D(x_real, y).mean()
            d_fake = self.D(x_fake, y).mean()
            gp     = self._gradient_penalty(x_real, x_fake, y)

            d_loss = d_fake - d_real + self.hparams.grad_penalty_weight * gp

            opt_d.zero_grad(set_to_none=True)
            self.manual_backward(d_loss)
            opt_d.step()

            self.log_dict(
                {
                    "d_loss": d_loss,
                    "gp_raw": gp,
                    "gp_w":   gp * self.hparams.grad_penalty_weight,
                    "wasserstein": d_real - d_fake,
                },
                prog_bar=True,
                on_step=True,
            )

        # ------------------------------------------------------------
        # 2. Train generator G (1 step)
        # ------------------------------------------------------------
        z = torch.randn(x_real.size(0), self.hparams.z_dim, device=self.device)
        x_fake = self.G(z, y)
        g_loss = -self.D(x_fake, y).mean()

        opt_g.zero_grad(set_to_none=True)
        self.manual_backward(g_loss)
        opt_g.step()

        self.log("g_loss", g_loss, prog_bar=True, on_step=True)
        return g_loss.detach()

    # --------------------------- sampling ---------------------------
    @torch.no_grad()
    def sample(self, label: int = 0, N: int = 16, z_dim: int = None):
        z_dim = z_dim or self.hparams.z_dim
        z = torch.randn(N, z_dim, device=self.device)
        y = torch.full((N,), label, dtype=torch.long, device=self.device)
        imgs = self.G(z, y)
        return (imgs + 1) / 2        # → [0,1]

    # --------------------------- optimizers -------------------------
    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.G.parameters(),
                                 lr=self.hparams.lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.D.parameters(),
                                 lr=0.5*self.hparams.lr, betas=(0.5, 0.999))
        return [opt_g, opt_d], []     # no schedulers
