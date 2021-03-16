import torchvision
import torchvision.transforms as transforms
from imgaug import augmenters as iaa 
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
## For loading CIFAR10 dataset and plotting 
from utils import *


# check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using {} to train'.format(device))

# Fix random seed 
same_seeds(0)

# Data augmentation
transform = transforms.Compose([
    iaa.Sequential([
        iaa.Crop(percent=(0,0.05)),
        iaa.Fliplr(0.2),
        iaa.GaussianBlur(sigma=(0, 0.5))
    ]).augment_image,
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transformVal = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

## Use class object to load image
class CIFAR10DataSet(Dataset):
    def __init__(self, image, label=None, transform=None):
        self.image = image 
        self.label = label
        self.transform = transform
    def __len__(self):
        return len(self.image) 
    def __getitem__(self, idx):
        if self.transform != None:
            return self.transform(self.image[idx]), self.label[idx]
        else:
            return self.image[idx], self.label[idx]

## Define model structure
class VGG16Classifier(nn.Module):
    def __init__(self, vgg16Model):
        super(VGG16Classifier, self).__init__()
        self.FeatureExtractor =  vgg16Model
        self.classifier = nn.Sequential(
            nn.Linear(1000,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )
    def forward(self, x):
        x = self.FeatureExtractor(x)
        return self.classifier(x)

## Load data from CIFAR10 dataset
device = torch.device("cuda" if torch.cuda.is_available else "cpu")
image_dir = '../data/cifar-10-batches-py'
X_train, Y_train, X_dev, Y_dev, X_test, Y_test = get_CIFAR10_data(image_dir, val_ratio=0.02)
trainset = CIFAR10DataSet(X_train,Y_train,transform=transform)
trainloader = DataLoader(trainset,batch_size=128,shuffle=True)
devset = CIFAR10DataSet(X_dev,Y_dev,transform=transformVal)
devloader = DataLoader(devset,batch_size=128,shuffle=False)
testset = CIFAR10DataSet(X_test,Y_test,transform=transformVal)
testloader = DataLoader(testset,batch_size=128,shuffle=False)
trainlen = trainset.__len__()
vallen = devset.__len__()
testlen = testset.__len__()

## Training argument
num_epoch = 30
lr = 0.001

## Import pretrained vgg16 model
vgg16 = models.vgg16(pretrained=True)
vgg16.to(device)
## Freezing featrue extractor's parameters
for param in vgg16.features.parameters():
    param.requires_grad = False
model = VGG16Classifier(vgg16).to(device)

# optimizer
param = list(model.FeatureExtractor.classifier.parameters())+list(model.classifier.parameters())
optimizer = torch.optim.Adam(param, lr=lr)
# loss function
criterion = nn.CrossEntropyLoss()
## Training
bestAcc = 0
train_loss , train_accuracy = [], []
val_loss , val_accuracy = [], []
for epoch in range(num_epoch):
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    print('(Training) Epoch {}'.format(epoch+1))
    for i, data in enumerate(trainloader):
        data, target = data[0].float().to(device), data[1].long().to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_running_loss += loss.item()
        _, preds = torch.max(output.data, 1)
        train_running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer.step()
    train_epoch_loss = train_running_loss/len(trainloader.dataset)
    train_epoch_accuracy = 100. * train_running_correct/len(trainloader.dataset)
    ## Validtion
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    print('(Validation) Epoch {}'.format(epoch+1))
    for i, data in enumerate(devloader):
        data, target = data[0].float().to(device), data[1].long().to(device)
        output = model(data)
        loss = criterion(output, target)
        val_running_loss += loss.item()
        _, preds = torch.max(output.data, 1)
        val_running_correct += (preds == target).sum().item()
    val_epoch_loss = val_running_loss/len(devloader.dataset)
    val_epoch_accuracy = 100. * val_running_correct/len(devloader.dataset)
    train_loss.append(train_epoch_loss)
    train_accuracy.append(train_epoch_accuracy)
    val_loss.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)
    print(f'Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_accuracy:.2f}')
    print(f'Validation Loss: {val_epoch_loss:.4f}, Validation Acc: {val_epoch_accuracy:.2f}')
    if val_epoch_accuracy > bestAcc:
        bestAcc = val_epoch_accuracy
        torch.save(model.state_dict(),'../model/VGG16model')

## Ploting training history
plt.figure(figsize=(10, 7))
plt.plot(range(1,num_epoch+1),train_accuracy, color='green', label='train accuracy',marker='o')
plt.plot(range(1,num_epoch+1),val_accuracy, color='blue', label='validataion accuracy',marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('../image/VGG16Accuracy.jpg')
plt.clf()

plt.figure(figsize=(10, 7))
plt.plot(range(1,num_epoch+1),train_loss, color='orange', label='train loss',marker='o')
plt.plot(range(1,num_epoch+1),val_loss, color='red', label='validataion loss',marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('../image/VGG16Loss.jpg')
plt.clf()

## Testing
model.eval()
test_running_loss = 0.0
test_running_correct = 0
predict = []
answer = []
prob = np.zeros((1,10),dtype=np.float)
print('(Testing)')
for i, data in enumerate(testloader):
    data, target = data[0].float().to(device), data[1].long().to(device)
    output = model(data)
    loss = criterion(output, target)
    test_running_loss += loss.item()
    _, preds = torch.max(output.data, 1)
    test_running_correct += (preds == target).sum().item()
    predict += list(preds.cpu().data.numpy())
    answer += list(target.cpu().data.numpy())
    prob = np.concatenate((prob,output.cpu().data.numpy()), axis=0)

prob = prob[1:]
prob_softmax = np.zeros((1,10))
for x in prob:
    prob_softmax = np.concatenate((prob_softmax,softmax(x).reshape(1,10)))
prob_softmax = prob_softmax[1:]

test_loss = test_running_loss/len(testloader.dataset)
test_accuracy = 100. * test_running_correct/len(testloader.dataset)
    
print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}')
np.save('../result/VGG16output', predict)
np.save('../result/VGG16target', answer)

## Calculate precision and recall for class 0 (airplane), 3 (cat), and 8 (ship)
calculatePrecisionRecall(predict,answer)
## Plotting confusion matrix 
plotConfusionMatrix(predict,answer)
## Plotting AUC curve
plotAUCCurve(prob_softmax,np.array(answer))