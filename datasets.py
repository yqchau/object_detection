import glob

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class SampleDataset(Dataset):
    def __init__(self, path, transform):
        self.transform = transform
        self.image_paths = glob.glob(f"{path}/images/*.png")
        self.label_paths = glob.glob(f"{path}/labels/*.csv")
        self.image_paths.sort()  # very important, haha..
        self.label_paths.sort()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        label_df = pd.read_csv(self.label_paths[idx])

        image = self.transform(image)
        label = label_df.values.astype(np.float64).squeeze()  # df -> np.float64
        return image, torch.FloatTensor(label).reshape(-1)


if __name__ == "__main__":
    import cv2
    import torchvision.transforms as transforms

    from visualize import annotate_bbox

    path = "./dataset/5_XYHW/train"
    image_size = 224
    n_samples = 5
    n_dim = 4
    transform = transforms.Compose(
        transforms=[
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    dataset = SampleDataset(path, transform=transform)
    x, y = dataset.__getitem__(0)

    img = x.view(3, image_size, image_size).permute(1, 2, 0).contiguous().numpy()
    params = y.view(n_samples, n_dim)
    annotate_bbox(img, image_size=image_size, params=params, type="XYHW")

    cv2.imshow("image", img)
    cv2.waitKey(2000)
