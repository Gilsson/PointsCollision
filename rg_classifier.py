import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import os
import torch


def f1_score(y_true, y_pred):
    """
    Compute the F1 score given the ground truth labels and predicted labels.
    Args:
    - y_true (torch.Tensor): Ground truth labels (binary tensor).
    - y_pred (torch.Tensor): Predicted labels (binary tensor).
    Returns:
    - f1 (float): Computed F1 score.
    """
    true_positives = torch.sum(y_true * y_pred).float()
    false_positives = torch.sum((1 - y_true) * y_pred).float()
    false_negatives = torch.sum(y_true * (1 - y_pred)).float()

    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)

    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    return f1


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Get a list of all image filenames in the root directory
        self.image_filenames = [
            filename for filename in os.listdir(root_dir) if filename.endswith(".png")
        ]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_filename = os.path.join(self.root_dir, self.image_filenames[idx])
        image = Image.open(img_filename).convert("RGB")

        # Extract the class label from the filename
        label = int(self.image_filenames[idx].split("_")[1].split(".")[0])

        if self.transform:
            image = self.transform(image)

        return image, label


import torch.nn.functional as F


def custom_scorer(y_true, y_pred):
    return


def train_model(model, train_loader, criterion, optimizer):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Compute binary cross entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Compute the focal term
        p_t = torch.exp(-bce_loss)  # Probability of correct classification
        focal_term = (1 - p_t) ** self.gamma

        # Compute the final loss
        loss = self.alpha * focal_term * bce_loss

        return torch.mean(loss)


class BackboneNetwork(nn.Module):
    def __init__(self, pretrained=True):
        super(BackboneNetwork, self).__init__()
        from torchvision import models

        self.efficientnet = models.efficientnet_b0(pretrained=True)

    def forward(self, x):
        # Extract features from the backbone network
        features = []
        x = self.efficientnet._swish(
            self.efficientnet._bn0(self.efficientnet._conv_stem(x))
        )
        features.append(x)
        idx = 0
        for block in self.efficientnet._blocks:
            drop_connect_rate = self.efficientnet._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.efficientnet._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if block._depthwise_conv.stride == [2, 2]:
                features.append(x)
            idx += 1
        return features


class FPN(nn.Module):
    def __init__(self, backbone_out_channels, out_channels=256):
        super(FPN, self).__init__()
        self.out_channels = out_channels
        self.lateral_convs = nn.ModuleList(
            [
                nn.Conv2d(channels, out_channels, kernel_size=1)
                for channels in backbone_out_channels[:-1]
            ]
        )
        self.smooth_convs = nn.ModuleList(
            [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
                for _ in range(len(backbone_out_channels) - 1)
            ]
        )
        self.topdown_convs = nn.ModuleList(
            [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
                for _ in range(len(backbone_out_channels) - 1)
            ]
        )
        self.topdown_convs.append(
            nn.Conv2d(backbone_out_channels[-1], out_channels, kernel_size=3, padding=1)
        )

    def forward(self, features):
        c2, c3, c4, c5 = features

        # Top-down pathway
        p = [self.topdown_convs[-1](c5)]
        for idx in range(len(self.topdown_convs) - 2, -1, -1):
            p_upsampled = nn.functional.interpolate(
                p[-1], scale_factor=2, mode="nearest"
            )
            p.append(self.topdown_convs[idx](p_upsampled))

        # Lateral connections and smooth layers
        p = p[::-1]
        for idx, (lateral_conv, smooth_conv) in enumerate(
            zip(self.lateral_convs, self.smooth_convs)
        ):
            lateral_feature = lateral_conv(features[idx])
            if idx > 0:
                lateral_feature = nn.functional.interpolate(
                    lateral_feature, scale_factor=2**idx, mode="nearest"
                )
            p[idx] += lateral_feature
            p[idx] = smooth_conv(p[idx])

        return tuple(p)


class GridBasedDetection(nn.Module):
    def __init__(self, num_classes, num_anchors_per_cell=3):
        super(GridBasedDetection, self).__init__()
        self.num_classes = num_classes
        self.num_anchors_per_cell = num_anchors_per_cell

        # Define layers for predicting bounding boxes and class probabilities
        self.bbox_conv = nn.Conv2d(
            256, num_anchors_per_cell * 4, kernel_size=3, padding=1
        )
        self.class_conv = nn.Conv2d(
            256, num_anchors_per_cell * num_classes, kernel_size=3, padding=1
        )

    def forward(self, features):
        # Predict bounding box offsets
        bbox_preds = self.bbox_conv(features)
        # Predict class probabilities
        class_preds = self.class_conv(features)

        # Reshape predictions
        batch_size, _, height, width = features.size()
        bbox_preds = (
            bbox_preds.permute(0, 2, 3, 1)
            .contiguous()
            .view(batch_size, height, width, self.num_anchors_per_cell, 4)
        )
        class_preds = (
            class_preds.permute(0, 2, 3, 1)
            .contiguous()
            .view(
                batch_size, height, width, self.num_anchors_per_cell, self.num_classes
            )
        )

        return bbox_preds, class_preds


class EfficientNetBasedMulticlass(nn.Module):
    def __init__(
        self,
        weights,
        num_classes=2,
        dropout=0.2,
    ):
        super(EfficientNetBasedMulticlass, self).__init__()
        from torchvision import models

        self.efficientnet = models.efficientnet_b3(weights=weights)
        self.fc = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        features = self.efficientnet.forward(x)
        features = features.flatten(start_dim=1)
        output = self.fc(features)
        output = self.softmax(output)
        return output


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout=0.2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding="same",
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding="same",
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(3, 3),
                    padding="same",
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.downsample(x) if self.downsample else x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x + identity)
        x = self.dropout(x)

        return x


class MultipleStepClassifier(nn.Module):
    def __init__(self, out_channels=3, dropout=0.2):
        super(MultipleStepClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 2, (3, 3)),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Conv2d(2, 4, (3, 3)),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 6, (3, 3)),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Conv2d(6, 8, (3, 3)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 10, (3, 3)),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10, 12, (3, 3)),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 14, (3, 3)),
            nn.BatchNorm2d(14),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(14, 16, (3, 3)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 18, (3, 3)),
            nn.BatchNorm2d(18),
            nn.ReLU(),
            nn.Conv2d(18, 20, (3, 3)),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Conv2d(20, 24, (3, 3)),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 28, (3, 3)),
            nn.BatchNorm2d(28),
            nn.ReLU(),
            nn.Conv2d(28, 32, (3, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, (3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, (3, 3)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(1),
            nn.Linear(15488, 2048),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.Dropout(dropout),
            nn.Linear(512, out_channels),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.network(x)


class OneStepClassifier(nn.Module):
    def __init__(self):
        super(OneStepClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(16, 16),
            ResidualBlock(16, 16),
            ResidualBlock(16, 16),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            ResidualBlock(16, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            ResidualBlock(32, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Flatten(1),
            nn.Dropout(p=0.2),
            # nn.Linear(1024, 1024),
            # nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.network(x)
        return x


class CollisionClassifier(nn.Module):
    def __init__(self, params):
        super(CollisionClassifier, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, 2, (3, 3)),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Conv2d(2, 4, (3, 3)),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 6, (3, 3)),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Conv2d(6, 8, (3, 3)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, (3, 3)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # ResidualBlock(8, 12),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # ResidualBlock(12, 16),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # ResidualBlock(16, 16),
            # ResidualBlock(16, 16),
            # ResidualBlock(12, 16),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # ResidualBlock(16, 32),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # ResidualBlock(16, 16),
            # ResidualBlock(16, 16),
            # ResidualBlock(16, 32),
            # ResidualBlock(128, 128),
            # ResidualBlock(128, 128),
            # ResidualBlock(128, 64),
            # nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # ResidualBlock(64, 64),
            # ResidualBlock(64, 64),
            # ResidualBlock(64, 32),
            # nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # ResidualBlock(32, 32),
            # ResidualBlock(32, 32),
            # ResidualBlock(32, 16),
            # nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # ResidualBlock(16, 16),
            # ResidualBlock(16, 16),
            # ResidualBlock(16, 16),
            # nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Flatten(1),
            # nn.Dropout(p=0.2),
            # nn.Linear(2048, 1024),
            nn.Linear(13456, 1024),
            nn.Dropout(params["dropout"]),
            nn.Linear(1024, 512),
            nn.Dropout(params["dropout"]),
            nn.Linear(512, 1),
            # nn.Dropout(params["dropout"]),
            # nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.network(x)
        return x

    @staticmethod
    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))


import torch.nn.functional as F
import torch.nn as nn


class HourglassBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HourglassBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        skip = x
        x = self.downsample(x)
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = nn.functional.relu(self.bn4(self.conv4(x)))
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        x += skip
        return x


class HourglassNetwork(nn.Module):
    def __init__(
        self,
        num_stages=2,
        num_channels=32,
        num_keypoints=28,
        dropout=0.5,
        weight_decay=1e-5,
    ):
        super(HourglassNetwork, self).__init__()
        self.num_stages = num_stages
        self.num_channels = num_channels
        self.init_conv = nn.Conv2d(3, num_channels, kernel_size=7, stride=2, padding=3)

        # Hourglass blocks
        self.hourglass_stages = nn.ModuleList(
            [self._build_hourglass_stage(num_channels) for _ in range(num_stages)]
        )

        # Keypoints head
        self.keypoints_head = nn.Conv2d(num_channels, num_keypoints, kernel_size=1)

        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.weight_decay = weight_decay

    def _build_hourglass_stage(self, num_channels):
        layers = []
        for _ in range(3):
            layers.append(HourglassBlock(num_channels, num_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial convolution
        x = nn.functional.relu(self.init_conv(x))

        intermediate_outputs = []
        # Hourglass blocks
        for stage in self.hourglass_stages:
            x = stage(x)
            intermediate_outputs.append(x)

        # Keypoints head
        keypoints = self.keypoints_head(x)
        return keypoints, intermediate_outputs


# Additional layers for post-processing
class PostProcessingModule(nn.Module):
    def __init__(self, input_size, output_size):
        super(PostProcessingModule, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Combined network
class CombinedNetwork(nn.Module):
    def __init__(
        self,
        params=None,
        num_stages=2,
        num_channels=16,
        num_keypoints=3,
        postprocessing_input_size=37632,
        postprocessing_output_size=3,
    ):
        super(CombinedNetwork, self).__init__()
        self.hourglass = HourglassNetwork(
            num_stages,
            num_channels,
            num_keypoints,
            dropout=params["dropout"] if params else 0.5,
            weight_decay=params["weight_decay"] if params else 1e-5,
        )
        self.postprocessing = PostProcessingModule(
            postprocessing_input_size, postprocessing_output_size
        )

    def forward(self, x):
        keypoints, intermediate_outputs = self.hourglass(x)
        processed_output = self.postprocessing(keypoints)
        return processed_output


class OptunaNet(nn.Module):
    def __init__(self, num_layers, num_filters, use_maxpool):
        super(OptunaNet, self).__init__()
        layers = []
        in_channels = 3
        for i in range(num_layers):
            layers.append(
                nn.Conv2d(in_channels, num_filters[i], kernel_size=3, padding=1)
            )
            layers.append(nn.BatchNorm2d(num_filters[i]))
            layers.append(nn.ReLU())
            if use_maxpool[i]:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = num_filters[i]
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Sequential(nn.Linear(num_filters[-1], 512), nn.Linear(512, 3))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x
