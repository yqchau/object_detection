import glob

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


def gaussian2D(shape, sigma=1):
    m, n = ((ss - 1.0) / 2.0 for ss in shape)
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[
        radius - top : radius + bottom, radius - left : radius + right
    ]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap


class SampleHeatMapDataset(Dataset):
    def __init__(self, path):
        self.image_size = 224
        self.downsample = 4
        self.transform = transforms.Compose(
            transforms=[
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        )
        self.image_paths = glob.glob(f"{path}/images/*.png")
        self.label_paths = glob.glob(f"{path}/labels/*.csv")
        self.image_paths.sort()
        self.label_paths.sort()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        label_df = pd.read_csv(self.label_paths[idx])

        image = self.transform(image)
        label = label_df.values.astype(np.float64).squeeze()  # df -> np.float64
        label *= self.image_size // self.downsample

        heatmap = np.zeros(
            (self.image_size // self.downsample, self.image_size // self.downsample)
        ).astype(float)
        for (x, y, D, _) in label:
            draw_umich_gaussian(
                heatmap,
                (x, y),
                int(D // (2 * self.downsample)),
            )
        # heatmap_tensor = torch.FloatTensor(heatmap).unsqueeze(0)
        return image, torch.FloatTensor(heatmap).unsqueeze(0)
