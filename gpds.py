
#%%
import os
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tqdm
#%%
maxsize = [1400, 2200]

datapath = '/shared/datasets/signatures/GPDS150'

def loadimages(path, ppl=150):
  arr = [[] for _ in range(ppl)]
  
  for imgpath in os.listdir(path):
    img = np.array(Image.open(path+imgpath).convert('L'))[:maxsize[0], :maxsize[1]]
    img = np.pad(img, ((0, maxsize[0] - img.shape[0]), (0, maxsize[1] - img.shape[1])))
    arr[int(imgpath.split()[0].split('-')[1])-1].append(img)
  return np.array(arr)

train_genuine = loadimages(datapath+'/train/genuine/')
train_forge = loadimages(datapath+'/train/forge/')

#%%

train_genuine.shape

#%%
def readpath(path):
  collection = {}
  t = tqdm.tqdm(os.listdir(path), desc=f'loading {path}')
  for img_path in t:
    if not img_path.endswith('.png'): continue
    id = img_path.split('_')[1]
    collection.setdefault(id, []).append(loadimage(path+'/'+img_path))
  res = np.array([np.array(collection[key]) for key in sorted(collection.keys())], dtype=np.float32) / 255.0
  return torch.nn.functional.adaptive_avg_pool2d(torch.Tensor(res), (200, 300),).to('cuda')



