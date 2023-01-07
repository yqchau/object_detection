import cv2
import torch

from datasets import SampleDataset
from model import CNN18


def annotate_bbox(img, image_size, params, type="XYHW"):
    for i in range(len(params)):
        center_x, center_y, W, H = params[i]
        center_x = int(center_x * image_size)
        center_y = int(center_y * image_size)
        W = int(W * image_size)
        H = int(H * image_size)

        color = [255, 255, 255]

        if type == "XYHW":
            img = cv2.rectangle(
                img,
                (center_x - W // 2, center_y - H // 2),
                (center_x + W // 2, center_y + H // 2),
                color,
            )
        elif type == "TLBR":
            img = cv2.rectangle(
                img,
                (center_x, center_y),
                (W, H),
                color,
            )
    return img


if __name__ == "__main__":
    n_samples = 1
    n_dim = 4
    image_size = 64

    model = CNN18(3, 4)  # FCNN()
    model.load_state_dict(torch.load("./weights.pth"))
    model.eval()
    dataset = SampleDataset(path="./dataset/1_XYHW/val")

    for i in range(80):
        x, y = dataset.__getitem__(i)
        y_pred = model(x.unsqueeze(0))

        img = x.view(3, image_size, image_size).permute(1, 2, 0).contiguous().numpy()
        params = y_pred.view(n_samples, n_dim)
        annotate_bbox(img, image_size=image_size, params=params, type="TLBR")

        cv2.imshow("image", img)
        cv2.waitKey(2000)