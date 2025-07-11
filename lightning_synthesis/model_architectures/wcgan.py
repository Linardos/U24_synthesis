import torch, torch.nn as nn, torch.nn.functional as F
import pytorch_lightning as pl

# ---------------------------------------------------------------------
#  1.   building blocks
# ---------------------------------------------------------------------
class ResBlockG(nn.Module):
    """Generator residual block with conditional BN."""
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch, affine=False)
        self.gn1 = nn.Linear(emb_dim, in_ch)           # γ
        self.gb1 = nn.Linear(emb_dim, in_ch)           # β
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)

        self.bn2 = nn.BatchNorm2d(out_ch, affine=False)
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
    def __init__(self, z_dim, emb_dim, n_classes, base_ch=64, out_size=64):
        super().__init__()
        self.embed = nn.Embedding(n_classes, emb_dim)
        self.fc    = nn.Linear(z_dim, 4*4*4*base_ch)
        ch = 4*base_ch
        self.blocks = nn.ModuleList([
            ResBlockG(ch, ch//2, emb_dim),    # 8×8
            ResBlockG(ch//2, ch//4, emb_dim), # 16×16
            ResBlockG(ch//4, ch//8, emb_dim), # 32×32
            ResBlockG(ch//8, ch//8, emb_dim)  # 64×64
        ])
        self.to_img = nn.Conv2d(ch//8, 1, 3, 1, 1)
        self.out_size = out_size

    def forward(self, z, y):
        y_emb = self.embed(y)
        h = self.fc(z).view(z.size(0), -1, 4, 4)
        for blk in self.blocks:
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            h = blk(h, y_emb)
        return torch.tanh(self.to_img(F.relu(h)))

class Discriminator(nn.Module):
    def __init__(self, n_classes, base_ch=64):
        super().__init__()
        ch = base_ch
        self.blocks = nn.ModuleList([
            ResBlockD(1, ch,   downsample=True),   # 32×32
            ResBlockD(ch, 2*ch, downsample=True),  # 16×16
            ResBlockD(2*ch, 4*ch, downsample=True),# 8×8
            ResBlockD(4*ch, 4*ch, downsample=False)
        ])
        self.linear = nn.utils.spectral_norm(nn.Linear(4*ch, 1))
        # Projection for class-conditioning (Miyato et al.)
        self.embed  = nn.utils.spectral_norm(nn.Embedding(n_classes, 4*ch))

    def forward(self, x, y):
        h = x
        for blk in self.blocks:
            h = blk(h)
        h = F.relu(h).sum(dim=(2, 3))          # global sum-pool
        out = self.linear(h) + (self.embed(y) * h).sum(dim=1, keepdim=True)
        return out.squeeze(1)                  # (B,)

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
    def __init__(self, n_classes, z_dim=128, lr=2e-4, n_critic=5,
                 grad_penalty_weight=10.0):
        super().__init__()
        self.save_hyperparameters()
        self.G = Generator(z_dim, emb_dim=128, n_classes=n_classes)
        self.D = Discriminator(n_classes)          # behaves as a critic (no σ)

        self.automatic_optimization = False        # manual opt loop
        torch.autograd.set_detect_anomaly(True)

    # --------------------------- utils ------------------------------
    def _gradient_penalty(self, real, fake, labels):
        with torch.autocast(device_type="cuda", enabled=False):
            ε = torch.rand(real.size(0), 1, 1, 1, device=self.device)
            interp = ε * real + (1 - ε) * fake
            interp.requires_grad_(True)

            interp_logits = self.D(interp, labels)
            grad = torch.autograd.grad(
                outputs=interp_logits, inputs=interp,
                grad_outputs=torch.ones_like(interp_logits),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            gp = (grad.view(grad.size(0), -1).norm(2, dim=1) - 1).pow(2).mean()
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

        self.log_dict({"d_loss": d_loss, "gp": gp,
                       "wasserstein": (d_real - d_fake)},
                      prog_bar=True, on_step=True)

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
                                 lr=self.hparams.lr, betas=(0.0, 0.9))
        opt_d = torch.optim.Adam(self.D.parameters(),
                                 lr=self.hparams.lr, betas=(0.0, 0.9))
        return [opt_g, opt_d], []     # no schedulers
