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

data_path = Path("./data_drones/Spectral_Augmented/Green_Channel")
image_path = data_path / ""

train_dir = image_path / "Train_Images"
test_dir = image_path / "Test_Images"

train_labels= pd.read_csv('data_drones/Spectral_Augmented/Labels/New_Train_Labels.csv')
test_labels = pd.read_csv('data_drones/Spectral_Augmented/Labels/New_Test_Labels.csv')

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

        if self.transform:
            img,target = self.transform(img,target)

        return img, target

class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        labels[0] = labels[0]*new_w/w
        labels[2] = labels[2]*new_w/w
        labels[1] = labels[1]*new_h/h
        labels[3] = labels[3]*new_h/h

        return {'image': img, 'labels': labels}
     
data_transforms = transforms.Compose([transforms.ToTensor()])

train_dataset = DronesPictureDataset(csv_file='data_drones/Spectral_Augmented/Labels/New_Train_Labels.csv',
                                   root=train_dir,
                                   transform=None)
test_dataset = DronesPictureDataset(csv_file='data_drones/Spectral_Augmented/Labels/New_Test_Labels.csv',
                                   root=test_dir,
                                   transform=None)

BATCH_SIZE = 8
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

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

from engine import train_one_epoch, evaluate

num_classes = 3
model = get_model_instance_segmentation(num_classes)
model.load_state_dict(torch.load('./fastRCNN/fastRCNN_GC_SGD_Aug.pth'))
model.eval()
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.1,
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
NUM_EPOCHS = 1

num_epochs = NUM_EPOCHS

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    #train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=1)
    
    #print("TRAINING DONE!")
    # update the learning rate
    #lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, train_dataloader, device=device)
    #torch.save(model.state_dict(), "./fastRCNN/fastRCNN_RGB_SGD.pth")

#torch.save(model.state_dict(), "./fastRCNN/fastRCNN_RGB_SGD_test.pth")
    

