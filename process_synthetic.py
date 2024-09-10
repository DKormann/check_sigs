#%%
import os
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tqdm


#%%

device="cuda:1"

maxsize = [1400, 2200]
datapath = '/shared/datasets/signatures/synthetic_gpds'

imgs = [p for p in os.listdir(datapath+"/002") if p.endswith('.jpg')]

def loadimage(path):
  img = (255- np.array(Image.open(path).convert('L')))[:maxsize[0], :maxsize[1]]
  return np.pad(img, ((0, maxsize[0] - img.shape[0]), (0, maxsize[1] - img.shape[1])))

def save_img(arr, name):
  img = Image.fromarray(arr.int().cpu().numpy().astype(np.uint8))
  img.save(f'{name}.jpg')

count = 10

for person_id in os.listdir(datapath):
  c = np.stack([loadimage(f'{datapath}/{person_id}/c-{person_id}-{i:0>2}.jpg') for i in range(1,25)])
  cf = np.stack([loadimage(f'{datapath}/{person_id}/cf-{person_id}-{i:0>2}.jpg') for i in range(1,31)])
  c = torch.nn.functional.adaptive_avg_pool2d(torch.tensor(np.array(c, np.float16), dtype=torch.float16), (200, 300))
  cf = torch.nn.functional.adaptive_avg_pool2d(torch.tensor(np.array(cf, np.float16), dtype=torch.float16), (200, 300))

  os.makedirs(f'{datapath}/cleaned/{person_id}', exist_ok=True)
  for i, arr in enumerate(c): save_img(arr, f'{datapath}/cleaned/{person_id}/c-{i}')
  for i, arr in enumerate(cf): save_img(arr, f'{datapath}/cleaned/{person_id}/cf-{i}')

  count -= 1
  if count == 0: break

#%%

path =  "/shared/datasets/signatures/preprocessed/forged/001/cf-001-30.jpg"


img = Image.open(path)
arr = np.array(img.convert('L'))



#%%

plt.imshow(c[4].cpu())
#%%

img = Image.fromarray(c.int().cpu().numpy())


#%%

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