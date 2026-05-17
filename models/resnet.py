"""
Model definitions for GOOD: Training-Free Guided Diffusion Sampling for OOD Detection.

Includes ResNet variants, ViT, ConvNeXt wrappers, and the ClassifierEnergy module
used for energy-based and feature-based guidance during diffusion sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
from torchvision.models import resnet18 as _tv_resnet18, ResNet as _TVResNet
from torchvision.transforms.functional import rgb_to_grayscale


# ---------------------------------------------------------------------------
# ResNetWithFeature: torchvision-based ResNet18 that exposes intermediate features.
# Used for medical / grayscale datasets where input has 1 channel.
# ---------------------------------------------------------------------------

class ResNetWithFeature(_TVResNet):
    """ResNet-18 variant that accepts single-channel (grayscale) images and
    exposes a ``features`` method returning the penultimate-layer embedding."""

    def __init__(self, num_classes: int = 10):
        super().__init__(
            block=models.resnet.BasicBlock,
            layers=[2, 2, 2, 2],
            num_classes=num_classes,
        )
        # Replace first conv to accept 1-channel input
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the feature vector before the final FC layer."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


# ---------------------------------------------------------------------------
# CIFAR-style ResNet blocks (3x3 conv1, no maxpool)
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super().__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super().__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        return out


class ResNet(nn.Module):
    """CIFAR-style ResNet (3x3 first conv, no maxpool)."""

    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out

    def forward_list(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out, torch.flatten(self.avgpool(out), 1)


# ---------------------------------------------------------------------------
# ViT / ConvNeXt feature wrappers
# ---------------------------------------------------------------------------

class VIT_features(nn.Module):
    """Wraps a torchvision ViT to expose its CLS token embedding."""

    def __init__(self, vit):
        super().__init__()
        self._process_input = vit._process_input
        self.class_token = vit.class_token
        self.encoder = vit.encoder
        self.heads = vit.heads

    def forward(self, x: torch.Tensor):
        x = self._process_input(x)
        n = x.shape[0]
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder(x)
        return x[:, 0]


class ConvNext_features(nn.Module):
    """Wraps a torchvision ConvNeXt to expose its global-average-pooled embedding."""

    def __init__(self, convnext):
        super().__init__()
        self.features = convnext.features
        self.avgpool = convnext.avgpool
        self.classifier = convnext.classifier

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        x = self.avgpool(x).view(x.size(0), -1)
        return x


# ---------------------------------------------------------------------------
# Factory functions & model registry
# ---------------------------------------------------------------------------

def _resnet18(device, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def _resnet34(device, **kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def _resnet50(device, **kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def _resnet101(device, **kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

def _convnext_base(device, **kwargs):
    return ConvNext_features(torchvision.models.convnext_base(weights="IMAGENET1K_V1"))

def _vit_b16(device, **kwargs):
    return VIT_features(torchvision.models.vit_b_16(weights="IMAGENET1K_V1").to(device))


model_dict = {
    "resnet18": [_resnet18, 512],
    "resnet34": [_resnet34, 512],
    "resnet50": [_resnet50, 2048],
    "resnet101": [_resnet101, 2048],
    "convnext_base": [_convnext_base, 1024],
    "vit_b16": [_vit_b16, 768],
}


# ---------------------------------------------------------------------------
# ResNet_Model: encoder + linear classifier
# ---------------------------------------------------------------------------

class ResNet_Model(nn.Module):
    """Encoder (from *model_dict*) followed by a linear classification head."""

    def __init__(self, name: str = "resnet50", num_classes: int = 10, device: str = "cuda"):
        super().__init__()
        model_fn, dim_in = model_dict[name]
        self.encoder = model_fn(device).to(device)
        self.fc = nn.Linear(dim_in, num_classes).to(device)

    def forward(self, x, vos=False):
        feat = self.encoder(x)
        logits = self.fc(feat)
        if vos:
            return feat, logits
        return logits

    def forward_repre(self, x):
        feat = self.encoder(x)
        return feat, self.fc(feat)

    def features(self, x):
        return self.encoder(x)

    def feature_list(self, x):
        if not hasattr(self.encoder, 'forward_list'):
            raise NotImplementedError(
                f"{type(self.encoder).__name__} does not support feature_list. "
                "Use ResNet-based architectures for Mahalanobis scoring."
            )
        out_list = []
        encoded_before, encoded = self.encoder.forward_list(x)
        out_list.append(encoded_before)
        return self.fc(encoded), out_list

    def intermediate_forward(self, x, layer_index):
        if not hasattr(self.encoder, 'forward_list'):
            raise NotImplementedError(
                f"{type(self.encoder).__name__} does not support intermediate_forward."
            )
        encoded_before, _ = self.encoder.forward_list(x)
        return encoded_before


# ---------------------------------------------------------------------------
# ClassifierEnergy: energy-score wrapper used during guided diffusion
# ---------------------------------------------------------------------------

class ClassifierEnergy(nn.Module):
    """Wraps a classifier to compute:
      - ``forward(x)`` -> negative free-energy  (logsumexp of logits)
      - ``features(x)`` -> penultimate-layer embedding

    Inputs are bilinearly resized to the resolution expected by the classifier
    (224 for ImageNet-100, 32 for CIFAR-100).
    """

    def __init__(self, network: str, num_classes: int, load: str, dataset: str, device: str):
        super().__init__()
        self.network = network
        self.device = device
        self.size = 224 if dataset == "imagenet100" else 32

        if network == "resnet18":
            self.model = ResNetWithFeature(num_classes=num_classes)
            self.model.load_state_dict(torch.load(load, map_location=device, weights_only=True))
        else:
            self.model = ResNet_Model(name=network, num_classes=num_classes, device=device)
            state = torch.load(load, map_location=device, weights_only=True)
            state = self._clean_state_dict(state, load)
            self.model.load_state_dict(state, strict=True)

        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _clean_state_dict(state_dict, load_path: str):
        """Remove 'module.' prefix and handle energy-checkpoint key mapping."""
        cleaned = {}
        for k, v in state_dict.items():
            new_k = k[7:] if k.startswith("module.") else k
            cleaned[new_k] = v

        if "energy" in load_path:
            remapped = {}
            for k, v in cleaned.items():
                if k.startswith("."):
                    remapped["encoder" + k] = v
                elif k == "ht":
                    remapped["fc.weight"] = v
                elif k == "":
                    remapped["fc.bias"] = v
                else:
                    remapped[k] = v
            return remapped
        return cleaned

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)

    def _maybe_grayscale(self, x: torch.Tensor) -> torch.Tensor:
        if self.network == "resnet18" and x.shape[1] == 3:
            x = rgb_to_grayscale(x, num_output_channels=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._maybe_grayscale(x)
        x = self._resize(x)
        logits = self.model(x)
        return torch.logsumexp(logits, dim=1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self._maybe_grayscale(x)
        x = self._resize(x)
        return self.model.features(x)
