
#%%
import os
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tqdm

import multiprocessing
#%%

device="cuda:1"

maxsize = [1400, 2200]
datapath = '/shared/datasets/signatures/GPDS150'

def loadimage(path):
  img = (255- np.array(Image.open(path).convert('L')))[:maxsize[0], :maxsize[1]]
  return np.pad(img, ((0, maxsize[0] - img.shape[0]), (0, maxsize[1] - img.shape[1])))

def loadimages(path, ppl=150):
  arr = [[] for _ in range(ppl)]
  for imgpath in os.listdir(path):
    img = loadimage(path+'/'+imgpath)
    arr[int(imgpath.split()[0].split('-')[1])-1].append(img)
  return torch.nn.functional.adaptive_avg_pool2d(torch.tensor(np.array(arr, np.float16), dtype=torch.float16) / 255.0, (200, 300))


# test set has the same people as train so we dont use that
train_genuine, test_genuine = loadimages(datapath+'/train/genuine/').split([120, 30]) 
train_forge, test_forge = loadimages(datapath+'/train/forge/').split([120, 30])

#%%

def getsample(n=50):
  labels = torch.randint(2, (n, 1, 1)).type(torch.bool)
  idxs = torch.randint(120, (n,))
  xn = torch.randint(16, (n,))
  x = train_genuine[idxs, xn]
  yn = torch.randint(16, (n,))
  y = torch.where(labels, train_genuine[idxs, yn], train_forge[idxs, yn])
  return x.to(device), y.to(device), labels.flatten().to(device)

x,y,labels = getsample(10)

#%%

