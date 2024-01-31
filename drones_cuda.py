import os
from pathlib import Path
import torch
import pandas as pd
#from skimage import io, transform
import numpy as np
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.functional import convert_image_dtype

from torch import nn
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

data_path = Path("./data_drones/RGB_Augmented/")
image_path = data_path / ""

train_dir = image_path / "Train_Images"
test_dir = image_path / "Test_Images"

train_labels= pd.read_csv('data_drones/RGB_Augmented/New_Train_Labels_RGB.csv')
test_labels = pd.read_csv('data_drones/RGB_Augmented/New_Test_Labels_RGB.csv')

def show_picture(image, labels):
    """Show image with landmarks"""
    plt.imshow(image)
    #plt.scatter([labels[0],labels[2],labels[0],labels[2]], [labels[1],labels[3],labels[3],labels[1]], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated
'''
plt.figure()
show_picture(io.imread(os.path.join(train_dir, img_name)),
               img_labels)
plt.show()
'''

class DronesPictureDataset(Dataset):

    def __init__(self,csv_file,root ,transform=None):
        self.img_labels         = pd.read_csv(csv_file)
        self.root               = root 
        self.transform          = transform
        self.imgs = list((os.listdir(os.path.join(root, ""))))

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self,idx):
        img_path = os.path.join(self.root,self.imgs[idx])
        img  = convert_image_dtype(read_image(img_path),dtype=torch.float32)

        #img = read_image(img_path)/255
        indices = self.img_labels[self.img_labels["filename"]==self.imgs[idx]].index.values
        
        boxes = torch.from_numpy(np.array(self.img_labels.iloc[indices,1:5]))

        #print(self.imgs[idx])
        #print(boxes)
        area = (boxes[:,3] - boxes[:,1]) * (boxes[:,2] - boxes[:,0])
        #labels = torch.tensor([int(x) for x in np.array(self.img_labels.iloc[indices,5])])
        #print(indices)
        labels = torch.from_numpy(np.array(self.img_labels.iloc[indices,5],dtype='int64'))
        #print(labels)
        
        #print(labels)
        #print(idx, "/" ,len(self.imgs))
        img = tv_tensors.Image(img)
        image_id = idx
        num_objs = len(indices)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes,format="XYXY",
                                                   canvas_size=F.get_size(img))
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transform is not None:
            img,target = self.transform(img,target)

        return img, target

from torchvision.transforms import v2 as T

def get_transforms(train):
    transforms = []
    if train:
       #transforms.append(T.ToImage())
       #transforms.append(T.RandomPhotometricDistort(p=1))
       #transforms.append(T.RandomZoomOut(fill={tv_tensors.Image:(123,117,104),"others":0}))
       transforms.append(T.RandomZoomOut(p=0.75))
       #transforms.append(T.RandomIoUCrop())
       transforms.append(T.RandomHorizontalFlip(p=0.75))
       transforms.append(T.RandomVerticalFlip(p=0.75))
       transforms.append(T.ClampBoundingBoxes())
       transforms.append(T.SanitizeBoundingBoxes())
    #transforms.append(T.ToDtype(torch.float,scale=True))
    #transforms.append(T.ToPureTensor())
    return T.Compose(transforms)
     

train_dataset = DronesPictureDataset(csv_file='data_drones/RGB_Augmented/New_Train_Labels_RGB.csv',
                                   root=train_dir,
                                   transform=get_transforms(train=True))
test_dataset = DronesPictureDataset(csv_file='data_drones/RGB_Augmented/New_Test_Labels_RGB.csv',
                                   root=test_dir,
                                   transform=None)

BATCH_SIZE = 24
NUM_WORKERS = os.cpu_count()

import utils
from utils import collate_fn

train_dataloader = DataLoader(dataset=train_dataset, 
                              batch_size=BATCH_SIZE, # how many samples per batch?
                              num_workers=NUM_WORKERS, # how many subprocesses to use for data loading? (higher = more)
                              shuffle=True,
                              collate_fn= collate_fn)

test_dataloader = DataLoader(dataset=test_dataset, 
                             batch_size=1, 
                             num_workers=NUM_WORKERS, 
                             shuffle=False,
                             collate_fn= collate_fn) 

from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

from torchvision.transforms import v2 as T

from engine import train_one_epoch, evaluate

num_classes = 3
model = get_model_instance_segmentation(num_classes)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.02,
    momentum=0.9,
    weight_decay=0.0005
)
#optimizer = torch.optim.Adam(params=model.parameters(), lr=0.050)

lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.3
)

# Set number of epochs
NUM_EPOCHS = 75

num_epochs = NUM_EPOCHS

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=1)
    
    print("TRAINING DONE!")
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, test_dataloader, device=device)
    evaluate(model, train_dataloader, device=device)
    #torch.save(model.state_dict(), "./fastRCNN/fastRCNN_RGB_SGD.pth")

torch.save(model.state_dict(), "./fastRCNN/fastRCNN_RGB_SGD_Aug_3.pth")
