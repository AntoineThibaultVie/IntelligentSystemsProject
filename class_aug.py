import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

data_path = Path("./")
image_path = data_path / "Blue_IR _train_test/"

train_dir = image_path / "train"
test_dir = image_path / "test"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

data_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.AutoAugment(),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder(root=train_dir, transform=data_transform, target_transform=None)
test_data = datasets.ImageFolder(root=test_dir, transform=transforms.Compose([transforms.Resize(size=(224, 224)), transforms.ToTensor()]))

def show_dataset(dataset, n=4):
    img = np.vstack((np.hstack((np.asarray(dataset[i][0]).transpose(1, 2, 0) for _ in range(n))) for i in range(n)))
    plt.imshow(img)
    plt.axis('off')
    plt.savefig('class_aug.png')  # Save the generated image

show_dataset(train_data)
plt.show()
