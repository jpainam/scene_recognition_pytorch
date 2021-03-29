#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/ashrutkumar/indoor-scene-recognition/blob/master/indoor_scene_recognition.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


#get_ipython().system('pip install torchsummary')


# In[2]:


import pandas as pd
import numpy as np
import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset,SubsetRandomSampler,Sampler
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
from torch import optim
import glob
from skimage import io, transform
from PIL import Image
import random
import PIL.ImageEnhance as ie
import copy
from torch.autograd import Variable
import PIL.Image as im
from math import floor
#from google.colab import files,drive
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
np.random.seed(1337)
random.seed(1337)


# In[4]:


class ImageDataset(Dataset): 
    
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame['Id'][idx])         
        image = Image.open(img_name).convert('RGB')                               
        #label = np.array(self.data_frame['Category'][idx])
        label = self.data_frame['Category'][idx]
        assert label is not None
        if self.transform:
            image = self.transform(image)                                         
        sample = (image, label)        
        return sample


# In[5]:


class FocalLoss(nn.Module):
    
    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
              
        # logit = F.softmax(input, dim=1)

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)

        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha
        #print(type(alpha),type(self.alpha))
        

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth/(self.num_class-1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha=alpha.to(device)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


# In[6]:


class SubsetSampler(Sampler):
     
    def __init__(self, indices):
        self.num_samples = len(indices)
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)
    
    def __len__(self):
        return self.num_samples


# Data Augmentation using various methods , source : https://github.com/mratsim/Amazon-Forest-Computer-Vision/blob/master/src/p_data_augmentation.py

# In[7]:


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()            .mul(alpha.view(1, 3).expand(3, 3))            .mul(self.eigval.view(1, 3).expand(3, 3))            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class Grayscale(object):

    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Contrast(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class RandomOrder(object):
    """ Composes several transforms together in random order.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        if self.transforms is None:
            return img
        order = torch.randperm(len(self.transforms))
        for i in order:
            img = self.transforms[i](img)
        return img


class ColorJitter(RandomOrder):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.transforms = []
        if brightness != 0:
            self.transforms.append(Brightness(brightness))
        if contrast != 0:
            self.transforms.append(Contrast(contrast))
        if saturation != 0:
            self.transforms.append(Saturation(saturation))

class RandomFlip(object):
    """Randomly flips the given PIL.Image with a probability of 0.25 horizontal,
                                                                0.25 vertical,
                                                                0.5 as is
    """
    
    def __call__(self, img):
        dispatcher = {
            0: img,
            1: img,
            2: img.transpose(im.FLIP_LEFT_RIGHT),
            3: img.transpose(im.FLIP_TOP_BOTTOM)
        }
    
        return dispatcher[random.randint(0,3)] #randint is inclusive

class RandomRotate(object):
    """Randomly rotate the given PIL.Image with a probability of 1/6 90°,
                                                                 1/6 180°,
                                                                 1/6 270°,
                                                                 1/2 as is
    """
    
    def __call__(self, img):
        dispatcher = {
            0: img,
            1: img,
            2: img,            
            3: img.transpose(im.ROTATE_90),
            4: img.transpose(im.ROTATE_180),
            5: img.transpose(im.ROTATE_270)
        }
    
        return dispatcher[random.randint(0,5)] #randint is inclusive
    
class PILColorBalance(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        alpha = random.uniform(1 - self.var, 1 + self.var)
        return ie.Color(img).enhance(alpha)

class PILContrast(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        alpha = random.uniform(1 - self.var, 1 + self.var)
        return ie.Contrast(img).enhance(alpha)


class PILBrightness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        alpha = random.uniform(1 - self.var, 1 + self.var)
        return ie.Brightness(img).enhance(alpha)

class PILSharpness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        alpha = random.uniform(1 - self.var, 1 + self.var)
        return ie.Sharpness(img).enhance(alpha)
    


class PowerPIL(RandomOrder):
    def __init__(self, rotate=True,
                       flip=True,
                       colorbalance=0.4,
                       contrast=0.4,
                       brightness=0.4,
                       sharpness=0.4):
        self.transforms = []
        if rotate:
            self.transforms.append(RandomRotate())
        if flip:
            self.transforms.append(RandomFlip())
        if brightness != 0:
            self.transforms.append(PILBrightness(brightness))
        if contrast != 0:
            self.transforms.append(PILContrast(contrast))
        if colorbalance != 0:
            self.transforms.append(PILColorBalance(colorbalance))
        if sharpness != 0:
            self.transforms.append(PILSharpness(sharpness))


# Utility Function to split dataset into train and validation

# In[8]:


def train_valid_split(dataset, test_size = 0.25, shuffle = False, random_seed = 0):
    length = dataset.__len__()
    indices = list(range(1,length))
    
    if shuffle == True:
        random.seed(random_seed)
        random.shuffle(indices)
    
    if type(test_size) is float:
        split = floor(test_size * length)
    elif type(test_size) is int:
        split = test_size
    else:
        raise ValueError('%s should be an int or a float' % str)
    return indices[split:], indices[:split]


# To calculate image statistics mean and standard deviation

# In[9]:


def calculate_img_stats_avg(loader):
    mean = 0.
    std = 0.
    nb_samples = 0.
    for imgs,_ in loader:
        batch_samples = imgs.size(0)
        imgs = imgs.view(batch_samples, imgs.size(1), -1)
        mean += imgs.mean(2).sum(0)
        std += imgs.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean,std


# In[10]:


transform_augmented = transforms.Compose([
        transforms.RandomResizedCrop(224),
        PowerPIL(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                     ])
transform_raw = transforms.Compose([
                     transforms.Resize((224,224)),
                     transforms.ToTensor(),
                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


# Downloading the data-set

# In[11]:


#!wget http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar


# Unzipping the Data-set

# In[12]:


#!tar -xvf /home/paul/datasets/indoorCVPR_09.tar


# Modifying the files, so that all images are in a common directory and creating a train.csv with the category numerically encoded

# In[13]:


'''
files=os.listdir("Images")
#print(files)
files.sort()
labeldict={}
count=0

for file in files:
    labeldict[file]=count
    count+=1

from collections import defaultdict
class_freq=defaultdict(int)
csvlist=[]

for file in files:
    if not os.path.isdir(os.path.join("Images", file)):
        continue
    images=os.listdir(os.path.join("Images", file))
    count=1
    for img in images:
        an=img.split(".")        
        newname=str(file)+"_"+str(count)+"."+str(an[-1])
        #os.copy(os.path.join("Images",file, img), os.path.join("Images",newname))
        shutil.copy(os.path.join("Images",file, img), os.path.join("Images",newname))
        csvlist.append([labeldict[file],newname])
        class_freq[file]+=1
        count+=1
        
#for file in files:
#    os.rmdir(file)
#print(csvlist)
import pandas as pd
df=pd.DataFrame(csvlist,columns=["Category","Id"])
df.to_csv("./Train.csv");
#!mv Train.csv ../Train.csv
'''

# Calculating weight of each category , for weighted sampling if needed

# In[14]:


'''from collections import defaultdict
class_freq=defaultdict(int)
freq=[]
for ind,val in df.iterrows():
    class_freq[int(val["Category"])]+=1
for ind,val in class_freq.items():
    freq.append([ind,val])
freq.sort()
freq=[i[1] for i in freq]
wt_per_class=[0.]*67
N=float(sum(freq))
for i in range(67):                                                   
        wt_per_class[i] = N/float(freq[i])
weight=[0]*len(df)
for ind,val in df.iterrows():
    cat=val["Category"]
    weight[ind]=wt_per_class[cat]'''


# Creating Train set and Validation set

# In[15]:


#trainset = ImageDataset(csv_file = './Train.csv', root_dir = './Images', transform=transform_augmented)
#valset   = ImageDataset(csv_file = './Train.csv', root_dir = './Images', transform=transform_raw)
#accset   = ImageDataset(csv_file = './Train.csv', root_dir = './Images', transform=transform_raw)

trainset = ImageFolder(root="./Images", transform=transform_augmented)
valset = ImageFolder(root="./Images", transform=transform_raw)
accset = ImageFolder(root="./Images", transform=transform_raw)


# Initialising train laoder and validation loader 

# In[26]:


train_idx, valid_idx = train_valid_split(trainset, 0.25)
#train_sampler = torch.utils.data.WeightedRandomSampler(weight, len(train_idx))
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetSampler(valid_idx)
train_loader = DataLoader(trainset,batch_size=64,sampler=train_sampler,num_workers=4)
valid_loader = DataLoader(valset,batch_size=64,sampler=valid_sampler,num_workers=4)
acc_loader   = DataLoader(accset,batch_size=64,num_workers=4)
#loader = DataLoader(calcset,batch_size=100)


# Getting the model architecture of the Resnext101_32x16d and the pre-trained weights on ImageNet 

# In[17]:


model_ft = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl')


# Copying some layers so that they can be re-assigned after freezing the model

# In[18]:


#avgpool = model_ft.avgpool
#l4 = model_ft.layer4
#l3 = model_ft.layer3


# Freezing the model

# In[19]:


for param in model_ft.parameters():
    param.requires_grad = False


# Customising the last FC layer as per our needs and assigning previously copied layers so that they are un-frozen

# In[20]:


model_ft.fc = nn.Sequential(nn.Dropout(p=0.5),
                            nn.Linear(2048, 1024),
                            nn.LeakyReLU(inplace=True),
                            nn.Linear(1024,67),
                            #nn.Softmax(dim=1),
                            )
#model_ft.layer4 = l4
#model_ft.layer3 = l3
#model_ft.avgpool = avgpool


# Sending the model to GPU

# In[29]:


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#!export CUDA_VISIBLE_DEVICES=0,1,2,3
model_ft = model_ft.cuda()


# Checking Model Summary

# In[30]:


#from torchsummary import summary
#summary(model_ft,input_size=(3,224,224))


# Defining the Loss function and optimizer

# In[23]:


criterion = nn.CrossEntropyLoss()
#criterion = FocalLoss()
optimizer = optim.SGD(model_ft.parameters(), lr=0.001,momentum=0.9)#,weight_decay=0.00005)


# Initialising best_acc and best_Acc weights

# In[24]:


best_acc=0.0
best_model_wts = copy.deepcopy(model_ft.state_dict())


# Training loop

# In[27]:


num_epochs=30 # insert suitable number ( 20-30), also manually lower learning rate
for epoch in range(num_epochs):    
    print("Epoch  : "+str(epoch))
    print("-"*10)
    
    #training loop
    running_loss = 0.0
    running_corrects=0
    wrong=0
    model_ft.train()
    for inp,labels in train_loader:
        inp=inp.cuda()
        labels=labels.cuda()
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model_ft(inp)            
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()       
    epoch_loss = running_loss / (len(train_loader)*1.0)
    print('TRAINING SET   Loss: {}'.format(epoch_loss))
    
    # validation loop
    if True:
        correct=0
        wrong=0
        model_ft.eval()
        for inp,labels in valid_loader:
            inp=inp.cuda()
            labels=labels.cuda()
            optimizer.zero_grad()
            with torch.no_grad():
                outputs = model_ft(inp)
            _, preds = torch.max(outputs.data, 1)
            correct += torch.sum(preds == labels)
            wrong += torch.sum(preds != labels)
                
        acc = (correct.float()) / ((correct+wrong).float())
        print('VALIDATION SET Correct: {} Wrong {} Acc {}'.format(correct,wrong,acc))
        if acc > best_acc:
            best_acc = acc
            best_model_wts = copy.deepcopy(model_ft.state_dict())      
        
            
    running_loss = 0.0
    running_correct = 0
model_ft.load_state_dict(best_model_wts)
torch.save( model_ft.state_dict(), "best_model_resnext_16d_2048_1024_dropout_0.5_c_wts.pkl")
print('------ Finished Training    -----')


# In[42]:


model_ft.load_state_dict(best_model_wts)


# Downloading the trained models weight, acc around 84%, trained with manual learning rate annealing anf un-freezing various layers(even cnn layers ) of the pre-trained model

# In[43]:


#get_ipython().system('pip install gdown')
#get_ipython().system('gdown https://drive.google.com/uc?id=1-2ayU2W8YnVgvfHEukbpT9-HyrLF5vXt')


# Loadind the weights of the downloaded model

# In[44]:


model_ft_wts=torch.load("best_model_resnext_16d_2048_1024_dropout_0.5_b.pkl")
model_ft.load_state_dict(model_ft_wts)


# In[45]:


#Checking Accuracy on Complete Data Set
if True: 
        correct=0
        wrong=0
        model_ft.eval()
        for inp,labels in acc_loader:
            inp=inp.cuda()
            labels=labels.cuda()
            optimizer.zero_grad()
            with torch.no_grad():
                outputs = model_ft(inp)
            _, preds = torch.max(outputs.data, 1)
            correct += torch.sum(preds == labels)
            wrong += torch.sum(preds != labels)
              
        acc = (correct.float()) / ((correct+wrong).float())
        print('ACCURACY SET   Correct: {} Wrong {} ACC {} '.format(correct,wrong,acc))


# In[ ]:




