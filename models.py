import torch
import torch.nn as nn
from torchvision import models
import yaml

# ── read number of classes from config ──────────────────────────────
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
num_classes = config.get('num_classes', 2)  # Default to 2 if not provided


# ── helper: replace first conv with 1-channel version, keep weights ─
def adapt_first_conv_to_grayscale(cnn: nn.Module) -> nn.Module:
    if isinstance(cnn, models.ResNet):
        old = cnn.conv1                               # (64,3,7,7)
        new = nn.Conv2d(1, old.out_channels,
                        kernel_size=old.kernel_size,
                        stride=old.stride,
                        padding=old.padding,
                        bias=False)
        new.weight.data = old.weight.data.mean(dim=1, keepdim=True)
        cnn.conv1 = new

    elif isinstance(cnn, models.DenseNet):
        old = cnn.features.conv0                      # (64,3,7,7)
        new = nn.Conv2d(1, old.out_channels,
                        kernel_size=old.kernel_size,
                        stride=old.stride,
                        padding=old.padding,
                        bias=False)
        new.weight.data = old.weight.data.mean(dim=1, keepdim=True)
        cnn.features.conv0 = new

    elif isinstance(cnn, models.EfficientNet):
        old = cnn.features[0][0]                      # (32,3,3,3)
        new = nn.Conv2d(1, old.out_channels,
                        kernel_size=old.kernel_size,
                        stride=old.stride,
                        padding=old.padding,
                        bias=False)
        new.weight.data = old.weight.data.mean(dim=1, keepdim=True)
        cnn.features[0][0] = new

    else:
        raise ValueError("Unsupported backbone for 1-channel adaptation")
    return cnn

# ── wrapper that replaces the classification head ───────────────────
class Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, p_drop: float = 0.5):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(p_drop)

        if hasattr(backbone, "fc"):                        # ResNet, EfficientNet
            in_feat = backbone.fc.in_features
            backbone.fc = nn.Linear(in_feat, num_classes)

        elif hasattr(backbone, "classifier"):              # DenseNet
            if isinstance(backbone.classifier, nn.Linear):
                in_feat = backbone.classifier.in_features
                backbone.classifier = nn.Linear(in_feat, num_classes)
            else:                                          # seq head
                in_feat = backbone.classifier[-1].in_features
                backbone.classifier[-1] = nn.Linear(in_feat, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return self.dropout(x)

# ── model factory ───────────────────────────────────────────────────
def get_model(model_name: str, num_classes: int = num_classes, pretrained: bool = True):
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

    elif model_name == "resnet101":
        weights = models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet101(weights=weights)

    elif model_name == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.densenet121(weights=weights)

    elif model_name == "densenet201":
        weights = models.DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.densenet201(weights=weights)

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

    else:
        raise ValueError(f"Unsupported backbone: {model_name}")

    # make it 1-channel but keep pretrained kernels
    backbone = adapt_first_conv_to_grayscale(backbone)

    # freeze early layers (optional but useful against over-fitting)
    for name, p in backbone.named_parameters():
        if not name.startswith(("layer3", "layer4", "features.6")):
            p.requires_grad = False

    return Classifier(backbone, num_classes)

# ── sanity check ────────────────────────────────────────────────────
if __name__ == "__main__":
    for name in ["resnet50", "densenet121", "densenet201", "efficientnet_b0"]:
        model = get_model(name)
        x = torch.randn(2, 1, 224, 224)
        y = model(x)
        print(f"{name:12s} output shape:", y.shape)
