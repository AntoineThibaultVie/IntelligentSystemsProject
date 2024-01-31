import os
from pathlib import Path
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.functional import convert_image_dtype

device = "cuda" if torch.cuda.is_available() else "cpu"

data_path = Path("./data_drones/RGB_Augmented/")
image_path = data_path / ""

train_dir = image_path / "Train_Images"
test_dir = image_path / "Test_Images"

train_labels = pd.read_csv('data_drones/RGB_Augmented/New_Train_Labels_RGB.csv')
test_labels = pd.read_csv('data_drones/RGB_Augmented/New_Test_Labels_RGB.csv')


def show_picture(image, labels, title):
    """Show image with bounding boxes"""
    img = image.permute(1, 2, 0).numpy()
    plt.imshow(img)
    for box in labels['boxes']:
        x, y, w, h = box.numpy()
        rect = plt.Rectangle((x, y), w - x, h - y, fill=False, edgecolor='red', linewidth=1)
        plt.gca().add_patch(rect)
    plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


class DronesPictureDataset(Dataset):
    def __init__(self, csv_file, root, transform=None):
        self.img_labels = pd.read_csv(csv_file)
        self.root = root
        self.transform = transform
        self.imgs = list((os.listdir(os.path.join(root, ""))))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        img = read_image(img_path).float()

        indices = self.img_labels[self.img_labels["filename"] == self.imgs[idx]].index.values
        boxes = torch.from_numpy(np.array(self.img_labels.iloc[indices, 1:5]))

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        labels = torch.from_numpy(np.array(self.img_labels.iloc[indices, 5], dtype='int64'))

        img = tv_tensors.Image(img)
        image_id = idx
        num_objs = len(indices)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img)),
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target


from torchvision.transforms import v2 as T


def get_transforms(train):
    transforms = []
    if train:
        transforms.append(T.RandomZoomOut(p=0.75))
        transforms.append(T.RandomHorizontalFlip(p=0.75))
        transforms.append(T.RandomVerticalFlip(p=0.75))
        transforms.append(T.ClampBoundingBoxes())
        transforms.append(T.SanitizeBoundingBoxes())
    return T.Compose(transforms)


train_dataset = DronesPictureDataset(csv_file='data_drones/RGB_Augmented/New_Train_Labels_RGB.csv',
                                     root=train_dir,
                                     transform=get_transforms(train=True))

test_dataset = DronesPictureDataset(csv_file='data_drones/RGB_Augmented/New_Test_Labels_RGB.csv',
                                    root=test_dir,
                                    transform=None)

# Display four sample images with bounding boxes
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for i in range(4):
    sample_img, sample_target = train_dataset[i]
    show_picture(sample_img, sample_target, f'Sample Image {i + 1}')
    axes[i // 2, i % 2].imshow(sample_img.permute(1, 2, 0).numpy())
    axes[i // 2, i % 2].axis('off')

plt.tight_layout()
plt.savefig('sample_images_with_boxes.png')  # Save the figure
plt.show()