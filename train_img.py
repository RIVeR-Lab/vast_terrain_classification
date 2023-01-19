#!/usr/bin/env python
# coding: utf-8

import os
import time
import copy
import tqdm
import torch
import typing
import torchvision
import numpy as np
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler

cudnn.benchmark = True
plt.ion()   # interactive mode

# Model Parameters
img_width = 512
img_height = 512
img_channels = 3
batch_size = 16
num_classes = 11

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(512),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(),
        
    ]),
    'val': transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = os.path.abspath('./data/labeled_data')
image_data = datasets.ImageFolder(data_dir, data_transforms['train'])
valid_size = 0.2 # Allocate 20% to training
test_size = 0.2 # Allocate 20% to testing
num_train = len(image_data)
indices = list(range(num_train))
split_train = int(np.floor((valid_size + test_size) * num_train))
split_val = int(np.floor(valid_size * num_train))
split_test = int(np.floor(test_size * num_train))

np.random.shuffle(indices)
test_idx = indices[:split_test]
valid_idx = indices[split_test:split_test+split_val]
train_idx = indices[split_train:]
print(f'Using: {len(train_idx)} training samples, {len(valid_idx)} validation samples, and {len(test_idx)} test samples')

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
test_sampler = SubsetRandomSampler(test_idx)

train_loader = torch.utils.data.DataLoader(
    image_data, batch_size=32, sampler=train_sampler,
    num_workers=8, pin_memory=True,
)
valid_loader = torch.utils.data.DataLoader(
    image_data, batch_size=32, sampler=valid_sampler,
    num_workers=8, pin_memory=True,
)


# Add in transformations
valid_loader.dataset.transform = data_transforms['val']
train_loader.dataset.transform = data_transforms['train']

dataloaders = {
    'train': train_loader,
    'val': valid_loader
}
dataset_sizes = {'train': len(train_sampler),'val': len(valid_sampler), 'test': len(test_sampler)}
class_names = list(image_data.class_to_idx.keys())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, scheduler, num_epochs=20):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    print(device)
    model.to(device)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm.tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    #outputs = model(inputs)
                    outputs = model(inputs)
                    preds = torch.argmax(outputs, dim=-1, keepdim=False)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), './weights/resnet18.wts')
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, test_loader, device):
    images_so_far = 0
    final_true = torch.tensor(())
    final_preds = torch.tensor(())
    correct = 0
    model.to(device)
    with torch.no_grad():
        for i, (inputs, labels) in tqdm.tqdm(enumerate(test_loader)):
            inputs = inputs.to(device)
            
            labels = labels.to(device)
            outputs = model(inputs)
            
            preds = torch.argmax(outputs, dim=-1, keepdim=False)
            assert preds.shape[0] == labels.shape[0], "wtf"
            
            images_so_far += inputs.shape[0]
            
            correct += (preds == labels).float().sum()
            
            final_true = torch.hstack((final_true, labels.cpu().flatten()))
            final_preds = torch.hstack((final_preds, preds.cpu().flatten()))
    
    return final_true, final_preds, (correct / images_so_far)

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, num_classes)
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=10)

model_ft.load_state_dict(torch.load('./weights/resnet18.wts'))
test_loader = torch.utils.data.DataLoader(
    image_data, batch_size=32, sampler=test_sampler,
    num_workers=8, pin_memory=True,
)

true, pred, acc = visualize_model(model_ft, test_loader, device)
print(f"Overall Accuracy: {float(acc)}")

# Create confusion matrix based on testing results
class_names_cap = list(map(lambda x: x.capitalize(), class_names))
plt.rcParams["font.family"] = "Times New Roman"
cm = confusion_matrix(true, pred)
# Normalize
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(cmn, annot=True, fmt='.3f', xticklabels=class_names_cap, yticklabels=class_names_cap, cmap="viridis")
plt.ylabel('Actual', fontsize=14)
plt.xlabel('Predicted', fontsize=14)
plt.xticks(rotation = 45, fontsize=12)
plt.yticks(rotation=45, fontsize=12) 
plt.title('ResNet-18 Confusion Matrix', fontsize=18)
plt.gcf().subplots_adjust(bottom=0.15)
plt.show(block=False)
