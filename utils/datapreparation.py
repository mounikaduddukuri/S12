import numpy as np
import torch
import os
import torchvision
import torchvision.transforms as transforms
import zipfile
import requests
from tqdm import notebook
from io import StringIO,BytesIO
from albumentations import PadIfNeeded, IAAFliplr, Compose, RandomCrop, Normalize, HorizontalFlip, Resize, ToFloat, Rotate, Cutout
from albumentations.pytorch import ToTensor
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, random_split
from PIL import Image

class album_Compose:
    def __init__(self, train=True):
        if train:
            self.albumentations_transform = Compose([
                                                     #PadIfNeeded(min_height=64, min_width=64, always_apply=True, p=1.0),
                                                     #RandomCrop(height=32, width=32, always_apply=True, p=1.0),
                                                     IAAFliplr(p=0.5),
                                                     Cutout(num_holes=1, max_h_size=8, max_w_size=8, p=0.5),
                                                     Normalize(mean=[0.49139968, 0.48215841, 0.44653091], std=[0.24703223, 0.24348513, 0.26158784],),
                                                     ToTensor()
                                                     ])
        else:
            self.albumentations_transform = Compose([
                                                     Normalize(mean=[0.49139968, 0.48215841, 0.44653091], std=[0.24703223, 0.24348513, 0.26158784],),
                                                     ToTensor()
                                                     ])

    def __call__(self, img):
        img = np.array(img)
        img = self.albumentations_transform(image=img)['image']
        return img

def load(datasetname, splitsize, split=False, albumentations=True):
  random_seed = 42
  
  if datasetname == 'cifar10':
    mean_tuple = (0.485, 0.456, 0.406)
    std_tuple = (0.229, 0.224, 0.225)
  elif datasetname == 'tinyimagenet':
    mean_tuple = (0.485, 0.456, 0.406)
    std_tuple = (0.229, 0.224, 0.225)

  if albumentations:
    train_transform = album_Compose(train=True)
    test_transform = album_Compose(train=False)
  else:
    # Transformation for Training
    train_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean_tuple, std_tuple)])
        # Transformation for Test
    test_transform = transforms.Compose([transforms.ToTensor(), 
                                       transforms.Normalize(mean_tuple, std_tuple)])

  if datasetname == 'cifar10':
    #Get the Train and Test Set
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  elif datasetname == 'tinyimagenet':
    down_url  = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    dataset = TinyImageNetDataset(down_url)
    classes = dataset.classes
    shuffle_dataset = True
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    print("Size of Dataset is: ", dataset_size)
    indices = list(range(dataset_size))
    split = int(np.floor(splitsize * dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)


# CUDA?
  cuda = torch.cuda.is_available()
  print("CUDA Available?", cuda)

    # For reproducibility
  torch.manual_seed(random_seed)

  if cuda:
    torch.cuda.manual_seed(random_seed) 

  if split:
    dataloader_args = dict(batch_size=512, num_workers=4, pin_memory=True) if cuda else dict(batch_size=64)
    trainloader = torch.utils.data.DataLoader(dataset,  **dataloader_args, sampler=train_sampler)
    testloader = torch.utils.data.DataLoader(dataset, **dataloader_args, sampler=valid_sampler)
  else:
    dataloader_args = dict(shuffle=True, batch_size=512, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
    trainloader = torch.utils.data.DataLoader(trainset, **dataloader_args)
    testloader = torch.utils.data.DataLoader(testset, **dataloader_args)
       

  return classes, trainloader, testloader

class TinyImageNetDataset(Dataset):
    def __init__(self, url):
        self.data = []
        self.target = []
        self.classes = []
        self.path = 'tiny-imagenet-200'
        
        self.download_dataset(url)
        
        wnids = open(f"{self.path}/wnids.txt", "r")
        for line in wnids:
          self.classes.append(line.strip())
        wnids.close()  

        wnids = open(f"{self.path}/wnids.txt", "r")
        
        for wclass in notebook.tqdm(wnids, desc='Loading Train Folder', total = 200):
          wclass = wclass.strip()
          for i in os.listdir(self.path+'/train/'+wclass+'/images/'):
            img = Image.open(self.path+"/train/"+wclass+"/images/"+i)
            npimg = np.asarray(img)
            if(len(npimg.shape) ==2):
              npimg = np.repeat(npimg[:, :, np.newaxis], 3, axis=2)
            self.data.append(npimg)  
            self.target.append(self.classes.index(wclass))

        val_file = open(f"{self.path}/val/val_annotations.txt", "r")
        for i in notebook.tqdm(val_file,desc='Loading Test Folder',total =10000):
          split_img, split_class = i.strip().split("\t")[:2]
          img = Image.open(f"{self.path}/val/images/{split_img}")
          npimg = np.asarray(img)
          if(len(npimg.shape) ==2):        
            npimg = np.repeat(npimg[:, :, np.newaxis], 3, axis=2)
          self.data.append(npimg)  
          self.target.append(self.classes.index(split_class))
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        target = self.target[idx]
        img = data     
        return data,target
    
    def classes(self):
      return self.classes    
    
    def download_dataset(self, url):
      if (os.path.isdir("tiny-imagenet-200")):
        print ('Images already downloaded...')
        return
      r = requests.get(url, stream=True)
      print ('Downloading TinyImageNet Data' )
      zip_ref = zipfile.ZipFile(BytesIO(r.content))
      for file in notebook.tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist())):
        zip_ref.extract(member = file)
      zip_ref.close()

#Old Album compose
#class album_Compose:
#    def __init__(self, train=True):
#        if train:
#            self.albumentations_transform = Compose([
#                                                     Rotate(limit=20, p=0.5),
#                                                     HorizontalFlip(),
#                                                     Cutout(num_holes=3, max_h_size=8, max_w_size=8, p=0.5),
#                                                     Normalize(mean=[0.49139968, 0.48215841, 0.44653091], std=[0.24703223, 0.24348513, 0.26158784],),
#                                                     ToTensor()
#                                                     ])
#        else:
#            self.albumentations_transform = Compose([
#                                                     Normalize(mean=[0.49139968, 0.48215841, 0.44653091], std=[0.24703223, 0.24348513, 0.26158784],),
#                                                     ToTensor()
#                                                     ])
#
#    def __call__(self, img):
#        img = np.array(img)
#        img = self.albumentations_transform(image=img)['image']
#        return img
#       

