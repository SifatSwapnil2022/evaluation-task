import json

import torch
import torch.nn as nn
from torchvision import models, transforms

import config


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SimpleCNN(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_model(arch, num_classes, pretrained=False):
    if arch == "simplecnn":
        return SimpleCNN(num_classes)
    weights = "DEFAULT" if pretrained else None
    if arch == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    raise ValueError(f"unknown arch: {arch}")


def train_transform(img_size):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD),
        transforms.RandomErasing(p=0.1),
    ])


def eval_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD),
    ])


def load_classes():
    path = config.MODELS_DIR / "classes.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Export class_names from the training notebook (see README).")
    return json.loads(path.read_text())


def load_trained(arch=None, weights=None, device="cpu"):
    arch = arch or config.ACTIVE_MODEL["arch"]
    weights = weights or config.ACTIVE_MODEL["weights"]
    classes = load_classes()
    path = config.MODELS_DIR / weights
    if not path.exists():
        raise FileNotFoundError(f"weights not found: {path}")
    model = build_model(arch, len(classes))
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device).eval(), classes
