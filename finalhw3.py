import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
import itertools
import matplotlib.pyplot as plt
import os
from PIL import Image


transform_crop_train = transforms.Compose([
    transforms.RandomResizedCrop(64),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])
transform_crop_test = transforms.Compose([
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

train_data = datasets.STL10(root='./data', split='train', download=True, transform=transform_crop_train)
test_data = datasets.STL10(root='./data', split='test', download=True, transform=transform_crop_test)
val = int(len(train_data)*0.15)
val_set, train_set = random_split(train_data, [val, len(train_data)-val])
y_data = np.array(train_data.labels)
classes = ['Airplane', 'Bird', 'Car', 'Cat', 'Deer', 'Dog', 'Horse', 'Monkey', 'Ship', 'Truck']
num_classes = len(classes)
cols = 4+ 1
plt.figure(figsize=(10, 12)) 
for y, cls in enumerate(classes):
    text_idx = y * cols + 1 
    plt.subplot(num_classes, cols, text_idx)
    plt.text(0.5, 0.5, cls, fontsize=12, ha='center', va='center', weight='bold')
    plt.axis('off')
    idxs = np.flatnonzero(y_data == y)
    idxs = np.random.choice(idxs, 4, replace=False)   
    for i, idx in enumerate(idxs):
        img_idx = y * cols + i + 2
        plt.subplot(num_classes, cols, img_idx)
        img = train_data.data[idx]
        img = np.transpose(img, (1, 2, 0))
        plt.imshow(img)
        plt.axis('off')

plt.tight_layout()
plt.show()

img_array = train_data.data[100]
img_array = np.transpose(img_array, (1, 2, 0))
orig_img = Image.fromarray(img_array)
aug_crop = transforms.RandomResizedCrop(64)
aug_flip = transforms.RandomHorizontalFlip(p=1.0)
aug_rot = transforms.RandomRotation(30)
img_cropped = aug_crop(orig_img)
img_flipped = aug_flip(orig_img)
img_rotated = aug_rot(orig_img)
fig, axs = plt.subplots(1, 4, figsize=(12, 4))
axs[0].imshow(orig_img)
axs[0].set_title("Original (96x96)")
axs[0].axis('off')
axs[1].imshow(img_cropped)
axs[1].set_title("Random Resized Crop (64x64)")
axs[1].axis('off')
axs[2].imshow(img_flipped)
axs[2].set_title("Random Horizontal Flip")
axs[2].axis('off')
axs[3].imshow(img_rotated)
axs[3].set_title("Random Rotation (30°)")
axs[3].axis('off')
plt.tight_layout()
plt.show()

def plot_and_save_results(train_losses, val_losses, train_accs, val_accs, model_name="Model_Results"):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange', linestyle='--')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Train Accuracy', color='green')
    plt.plot(epochs, val_accs, label='Validation Accuracy', color='red', linestyle='--')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    filename = f"{model_name}.png"
    plt.savefig(filename, dpi=300)
    #plt.show()
    plt.close()

#
def train(model, optimizer, Epochs, batch_size, model_name="Model"):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    criterion = nn.CrossEntropyLoss()
    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    for epoch in range(Epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0    
        for images, labels in train_loader:
            optimizer.zero_grad()
            images = images.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            labels = labels.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()  
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        avg_val_loss, val_acc = calculate_accuracy(val_loader, model, criterion)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Epoch [{epoch+1}/{Epochs}] | Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.2f}%")
    _, test_acc = calculate_accuracy(test_loader, model)
    print(f"Test Accuracy: {test_acc:.2f}%")
    return test_acc, train_losses, val_losses, train_accs, val_accs

def calculate_accuracy(loader, model, criterion=nn.CrossEntropyLoss()):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            labels = labels.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        avg_loss = running_loss / len(loader)
        accuracy = 100 * correct / total
    return avg_loss, accuracy
    
def getTop3Worst3(all_results):
    sorted_results = sorted(all_results, key=lambda x: x['acc'], reverse=True)
    top_3_models = sorted_results[:3]
    worst_3_models = sorted_results[-3:]
    #top3
    for i, res in enumerate(top_3_models):
        plot_and_save_results(
            res['train_losses'], res['val_losses'],
            res['train_accs'], res['val_accs'],
            model_name=f"Top_{i+1}_{res['name']}"
        )
    #lower3
    for i, res in enumerate(worst_3_models):
        plot_and_save_results(
            res['train_losses'], res['val_losses'],
            res['train_accs'], res['val_accs'],
            model_name=f"Worst_{i+1}_{res['name']}"
        )
def getOptimizer(opt_name, model_parameters, lr_val, reg):
    if opt_name == 'SGD':
        return optim.SGD(model_parameters, lr=lr_val, weight_decay=reg, momentum=0.9)
    elif opt_name == 'Adam':
        return optim.Adam(model_parameters, lr=lr_val, weight_decay=reg)
    elif opt_name == 'RMSprop':
        return optim.RMSprop(model_parameters, lr=lr_val, weight_decay=reg)
# Regression Model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        return out
#itrater
lr = [0.01, 0.001, 0.0001]
optimizers = ['SGD', 'Adam', 'RMSprop']
batch_sizes = [32, 64]
regularizations = [0.0, 0.001]
configs128 = list(itertools.product(lr, optimizers, batch_sizes, regularizations, [128, 256]))
configs256 = list(itertools.product(lr, optimizers, batch_sizes, regularizations, [256, 512]))
# Regression

all_results = []
for lr_val, opt_name, batch_size, reg , _ in configs128:
    model = LogisticRegressionModel((3*64*64), num_classes)
    optimizer = getOptimizer(opt_name, model.parameters(), lr_val, reg)
    Epochs = 10
    file_name = f"Reggression_LR{lr_val}_Opt{opt_name}_BS{batch_size}_Regulizer{reg}"
    test_acc, t_losses, v_losses, t_accs, v_accs = train(model, optimizer, Epochs, batch_size=batch_size, model_name=file_name)
    all_results.append({
        'name': file_name,
        'acc': test_acc,
        'train_losses': t_losses,
        'val_losses': v_losses,
        'train_accs': t_accs,
        'val_accs': v_accs
    })
getTop3Worst3(all_results)



# Fnn 3layers
all_results = []
top_acc = 0
for lr_val, opt_name, batch_size, reg, layer_size in configs256:
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3*64*64, layer_size),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.BatchNorm1d(layer_size),
        nn.Linear(layer_size, layer_size//2),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.BatchNorm1d(layer_size//2),
        nn.Linear(layer_size//2, layer_size//4),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.BatchNorm1d(layer_size//4),
        nn.Linear(layer_size//4, num_classes)
    )
    optimizer = getOptimizer(opt_name, model.parameters(), lr_val, reg)
    Epochs = 10
    file_name = f"FullyConnected3layrs_LR{lr_val}Opt{opt_name}BS{batch_size}Regulizatio{reg}_LS{layer_size}_amountlayers3"
    test_acc, t_losses, v_losses, t_accs, v_accs = train(model, optimizer, 10,batch_size=batch_size, model_name=file_name)
    all_results.append({
        'name': file_name,
        'acc': test_acc,
        'train_losses': t_losses,
        'val_losses': v_losses,
        'train_accs': t_accs,
        'val_accs': v_accs
    })
getTop3Worst3(all_results)
# 4 layers
all_results = []
for lr_val, opt_name, batch_size, reg, layer_size in configs256:
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3*64*64, layer_size),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.BatchNorm1d(layer_size),
        nn.Linear(layer_size, layer_size//2),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.BatchNorm1d(layer_size//2),
        nn.Linear(layer_size//2, layer_size//4),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.BatchNorm1d(layer_size//4),
        nn.Linear(layer_size//4, layer_size//8),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.BatchNorm1d(layer_size//8),
        nn.Linear(layer_size//8, num_classes)
    )
    optimizer = getOptimizer(opt_name, model.parameters(), lr_val, reg)
    file_name = f"fullyconnected4layers_LR{lr_val}Opt{opt_name}BS{batch_size}Reg{reg}LS{layer_size}"
    test_acc, t_losses, v_losses, t_accs, v_accs = train(model, optimizer, 10, batch_size=batch_size, model_name=file_name)
    all_results.append({
        'name': file_name,
        'acc': test_acc,
        'train_losses': t_losses,
        'val_losses': v_losses,
        'train_accs': t_accs,
        'val_accs': v_accs
    })
getTop3Worst3(all_results)

#Convolutional Neural Network Model
all_results = []
for lr_val, opt_name, batch_size, reg, layer_size in configs128:
    model = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(128, 256, kernel_size=3, padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Flatten(),
    nn.Linear(256 * 8 * 8, layer_size),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(layer_size, layer_size//2),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(layer_size//2, num_classes)
    )
    optimizer = getOptimizer(opt_name, model.parameters(), lr_val, reg)
    Epochs = 10
    file_name = f"ConvelutionNN_LR{lr_val}Opt{opt_name}BS{batch_size}Regurlization{reg}LS{layer_size}"
    test_acc, t_losses, v_losses, t_accs, v_accs =train(model, optimizer, Epochs,batch_size=batch_size, model_name=file_name)
    all_results.append({
        'name': file_name,
        'acc': test_acc,
        'train_losses': t_losses,
        'val_losses': v_losses,
        'train_accs': t_accs,
        'val_accs': v_accs
    })
getTop3Worst3(all_results)

# MobileNetV2 Model frozen
all_results = []
for lr_val, opt_name, batch_size, reg, layer_size in configs128:
    model = models.mobilenet_v2(weights='DEFAULT')
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, layer_size),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(layer_size, layer_size//2),
        nn.ReLU(),
        nn.Linear(layer_size//2, num_classes)
    )
    for param in model.features.parameters():
        param.requires_grad = False
    optimizer = getOptimizer(opt_name, model.parameters(), lr_val, reg)
    Epochs = 10
    file_name = f"MobileV2_LR{lr_val}Opt{opt_name}BS{batch_size}Reg{reg}LS{layer_size}"
    test_acc, t_losses, v_losses, t_accs, v_accs =train(model, optimizer, Epochs,batch_size=batch_size, model_name=file_name)
    all_results.append({
        'name': file_name,
        'acc': test_acc,
        'train_losses': t_losses,
        'val_losses': v_losses,
        'train_accs': t_accs,
        'val_accs': v_accs
    })
getTop3Worst3(all_results)

# MobileNetV2 finetune
all_results =[]
for lr_val, opt_name, batch_size, reg, layer_size in configs128:
    model = models.mobilenet_v2(weights='DEFAULT')
    for param in model.features.parameters():
        param.requires_grad = True
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(in_features, layer_size),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(layer_size, layer_size//2),
    nn.ReLU(),
    nn.Linear(layer_size//2, num_classes)
    ) 
    optimizer = getOptimizer(opt_name, model.parameters(), lr_val, reg)
    Epochs = 10
    file_name = f"MobileNetV2_FT_LR{lr_val}Opt{opt_name}BS{batch_size}Regulization{reg}LS{layer_size}"
    test_acc, t_losses, v_losses, t_accs, v_accs =train(model, optimizer, Epochs,batch_size=batch_size, model_name=file_name)
    all_results.append({
        'name': file_name,
        'acc': test_acc,
        'train_losses': t_losses,
        'val_losses': v_losses,
        'train_accs': t_accs,
        'val_accs': v_accs
    })
getTop3Worst3(all_results)
