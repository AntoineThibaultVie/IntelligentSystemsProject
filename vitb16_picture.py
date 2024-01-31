import os
from pathlib import Path
from re import T

data_path = Path("./")
image_path = data_path / "Blue_IR _train_test/"

train_dir = image_path / "train"
test_dir = image_path  / "test"

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from torchvision.io import read_image


from torch import nn

import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

data_transform = transforms.Compose([
    # Resize the images to 64x64
    transforms.Resize(size=(224, 224)),
    # Flip the images randomly on the horizontal
    # transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
    # transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.AutoAugment(),
    # Turn the image into a torch.Tensor
    transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
])

train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                  transform=data_transform, # transforms to perform on data (images)
                                  target_transform=None) # transforms to perform on labels (if necessary)

test_data = datasets.ImageFolder(root=test_dir, 
                                 transform=transforms.Compose([transforms.Resize(size=(224, 224)),transforms.ToTensor()]))


from torch.utils.data import DataLoader

# Setup batch size and number of workers 
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()

train_dataloader = DataLoader(dataset=train_data, 
                              batch_size=BATCH_SIZE, # how many samples per batch?
                              num_workers=NUM_WORKERS, # how many subprocesses to use for data loading? (higher = more)
                              shuffle=True) # shuffle the data?

test_dataloader = DataLoader(dataset=test_data, 
                             batch_size=1, 
                             num_workers=NUM_WORKERS, 
                             shuffle=False) # don't usually need to shuffle testing data


def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer):
    # Put model in train mode
    model.train()
    
    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X.to(device))

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

        print("Batch progress:",batch,"/",len(dataloader))
        if batch % 10 == 0 and batch != 0:
            print("################################")
            print("Training: ",round(batch/len(dataloader)*100),"%")
            print("################################")
            
    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module):
    # Put model in eval mode
    model.eval() 
    
    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)
    
            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            
            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def confusion_matrix(pred,target,num_classes):
    confmat = torch.zeros([num_classes,num_classes])
    for i in range(len(target)):
        if pred[i] != target[i]:
            confmat[pred[i],target[i]] += 1
    return confmat 

    
def test_confusion_map(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module):
    model.eval()
    test_loss,test_acc = 0,0
    num_classes = 12
    confmat = torch.zeros([num_classes,num_classes])
    with torch.inference_mode():
        for batch,(X,y) in enumerate(dataloader):
            X,y = X.to(device),y.to(device)
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits,y)
            test_loss += loss.item()
            test_pred_labels = test_pred_logits.argmax(dim=1)
            # preds += test_pred_labels
            # targets += y
            confmat += confusion_matrix(test_pred_labels,y,num_classes)
            test_acc += ((test_pred_labels==y).sum().item()/len(test_pred_labels))

    test_loss = test_loss/len(dataloader)
    test_acc = test_acc/len(dataloader)
    print(f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )
    print(confmat)
    
from tqdm.auto import tqdm

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):
    
    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss, test_acc = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn)
        
        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # 6. Return the filled results at the end of the epochs
    return results

# Set random seeds
torch.manual_seed(42) 
torch.cuda.manual_seed(42)

# Set number of epochs
NUM_EPOCHS = 100

from torchvision.models import vit_b_16, ViT_B_16_Weights
# Recreate an instance of TinyVGG

model = vit_b_16()
model.load_state_dict(torch.load('./vitb16/vitb16_cuda.pth',map_location=device))
# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

# Start the timer
from timeit import default_timer as timer 
start_time = timer()

#test_confusion_map(model=model,
#                   dataloader=test_dataloader,
#                   loss_fn=loss_fn)
# End the timer and print out how long it took
end_time = timer()
print(f"Total test time: {end_time-start_time:.3f} seconds")


image_diseased_train = read_image("./Blue_IR _train_test/train/Tomato_Diseased/IMG_8680.jpg")
image_healthy_train = read_image("./Blue_IR _train_test/train/Tomato_Healthy/IMG_6955.jpg")
image_diseased_test = read_image("./Blue_IR _train_test/test/Potato_diseased/IMG_4413.jpg")
image_healthy_test = read_image("./Blue_IR _train_test/test/Potato_healthy/IMG_4114.jpg")

classes = ('Avocado_Diseased', 'Avocado_Healthy',
           'Lime_Diseased', 'Lime_Healthy', 'Potato_diseased', 'Potato_healthy',
             'Tomato_Diseased',"Tomato_Healthy")
classes_list = ['Avocado_Diseased', 'Avocado_Healthy',
           'Lime_Diseased', 'Lime_Healthy', 'Potato_diseased', 'Potato_healthy',
             'Tomato_Diseased',"Tomato_Healthy"]

dataiter = iter(test_dataloader)
images, labels = next(dataiter)
outputs = model(images)


correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_dataloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
print(f'Accuracy of the network on the test images: {100 * correct / total:.3f} %')

# prepare to count predictions for each class
classes = ('Avocado_Diseased', 'Avocado_Healthy',
           'Lime_Diseased', 'Lime_Healthy', 'Potato_diseased', 'Potato_healthy',
             'Tomato_Diseased',"Tomato_Healthy")

correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in test_dataloader:
        images, labels = data
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1
    

# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.3f} %')


with torch.no_grad():
    i = 0
    for data in test_dataloader:
        sample_image, sample_label = data
        sample_output = model(sample_image)
        _, predictions = torch.max(sample_output, 1)

        # Assuming your input image is in range [0, 1]
        if i == 20:
            image = (255.0 * sample_image.squeeze().cpu().numpy()).astype(np.uint8)

            if image.shape[0] == 3:
                image = image.transpose(1, 2, 0)

                fig, ax = plt.subplots(figsize=(12, 12))
                ax.imshow(image)
                
                annotation_text = f"Predicted: {predictions.item()}: {classes_list[predictions[0]]}"
                ax.text(
                0.5, 0.95, annotation_text,
                horizontalalignment='center',
                verticalalignment='center',
                transform=plt.gca().transAxes,
                color='red', fontsize=12
                )
                plt.show()
                fig.savefig('potato_diseased_vitb16.png')
                print(predictions)
        if i == 28:
            image = (255.0 * sample_image.squeeze().cpu().numpy()).astype(np.uint8)

            if image.shape[0] == 3:
                image = image.transpose(1, 2, 0)

                fig, ax = plt.subplots(figsize=(12, 12))
                ax.imshow(image)
                
                annotation_text = f"Predicted: {predictions.item()}: {classes_list[predictions[0]]}"
                ax.text(
                0.5, 0.95, annotation_text,
                horizontalalignment='center',
                verticalalignment='center',
                transform=plt.gca().transAxes,
                color='red', fontsize=12
                )
                plt.show()
                fig.savefig('potato_healthy_vitb16.png')
                print(predictions)
            
        i += 1






