import os
import random

import cv2
import numpy as np
import pandas as pd


def sample_single_data(
    image,
    margin=20,
    annotation_type="XYHW",
    annotation_color=(200, 100, 200),
):
    image_x, image_y = image.shape[0:2]
    center_x = random.randint(margin, image_x - margin)
    center_y = random.randint(margin, image_y - margin)
    radius = random.randint(image_x // 20, image_y // 10)

    image = cv2.circle(image, (center_x, center_y), radius, annotation_color, -1)

    if annotation_type == "XYHW":
        return image, [
            center_x / 224.0,
            center_y / 224.0,
            radius * 2 / 224.0,
            radius * 2 / 224.0,
        ]
    elif annotation_type == "TLBR":
        return image, [
            (center_x - radius) / 224.0,
            (center_y - radius) / 224.0,
            (center_x + radius) / 224.0,
            (center_y + radius) / 224.0,
        ]


def sample_data(
    num_data=800,
    num_sample=1,
    image_size=(224, 224, 3),
    annotation_type="XYHW",
    dataset_path="./dataset/1_XYHW/train",
    label_columns=["center_x", "center_y", "width", "height"],
    annotation_color=(200, 100, 200),
):
    for i in range(num_data):
        num_sample = 5
        image = np.zeros(image_size, np.uint8)
        labels = []
        for _ in range(num_sample):
            image, label = sample_single_data(
                image,
                margin=image_size[0] // 10,
                annotation_type=annotation_type,
                annotation_color=annotation_color,
            )
            labels.append(label)

        labels = pd.DataFrame(labels, columns=label_columns)

        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
            os.mkdir(dataset_path + "/images")
            os.mkdir(dataset_path + "/labels")

        cv2.imwrite(f"{dataset_path}/images/{i:04d}.png", image)
        labels.to_csv(f"{dataset_path}/labels/{i:04d}.csv", index=False)


if __name__ == "__main__":
    train_num_data = 800
    train_dataset_path = "./dataset/1_TEST/train"

    val_num_data = 80
    val_dataset_path = "./dataset/1_TEST/val"

    num_sample = 1
    image_size = (224, 224, 3)
    annotation_type = "TLBR"  # XYHW or TLBR (Top Left Bottom Right coordinates)
    label_columns = ["center_x", "center_y", "width", "height"]
    annotation_color = (200, 100, 200)

    sample_data(
        num_data=train_num_data,
        num_sample=num_sample,
        image_size=image_size,
        annotation_type=annotation_type,
        dataset_path=train_dataset_path,
        label_columns=label_columns,
        annotation_color=annotation_color,
    )
    sample_data(
        num_data=val_num_data,
        num_sample=num_sample,
        image_size=image_size,
        annotation_type=annotation_type,
        dataset_path=val_dataset_path,
        label_columns=label_columns,
        annotation_color=annotation_color,
    )
