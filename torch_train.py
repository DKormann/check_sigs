#%%
import os
import torch
from torch import nn
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import tqdm
#%%

maxdims = [576, 672]
datapath = '/shared/datasets/signatures/signatures'

def loadimage(path):
  img = Image.open(path).convert('L')
  img = np.array(img)
  img = img[:maxdims[0], :maxdims[1]]
  img = np.pad(img, ((0, maxdims[0] - img.shape[0]), (0, maxdims[1] - img.shape[1])))
  return img

def readpath(path):
  collection = {}
  t = tqdm.tqdm(os.listdir(path), desc=f'loading {path}')
  for img_path in t:
    if not img_path.endswith('.png'): continue
    id = img_path.split('_')[1]
    collection.setdefault(id, []).append(loadimage(path+'/'+img_path))
  res = np.array([np.array(collection[key]) for key in sorted(collection.keys())], dtype=np.float32) / 255.0
  return torch.nn.functional.adaptive_avg_pool2d(torch.Tensor(res), (200, 300),).to('cuda')

org = readpath(datapath+'/full_org')
forg = readpath(datapath+'/full_forg')

train_org = org[:50]
train_forg = forg[:50]
test_org = org[50:]
test_forg = forg[50:]


#%%
def getsample(n=50):
  labels = np.random.randint( 2, size=(n,1,1))
  idxs = np.random.randint(50, size=n)
  xn = np.random.randint(24, size=n)
  x = train_org[idxs, xn]
  yn = np.random.randint(24, size=n)

  labels = torch.tensor(labels, dtype=bool, device='cuda')
  y = torch.where(labels, train_org[idxs, yn], train_forg[idxs, yn])
  return x, y, labels.flatten()

x,y,labels = getsample(10)


x.shape # torch.Size([10, 200, 300])
#%%

class Model(nn.Module):

  def __init__(self):
    super(Model, self).__init__()
    self.blocks = nn.Sequential(
      nn.Conv2d(1, 16, 11), nn.ReLU(), nn.Conv2d(16, 32, 5 ),
      nn.LayerNorm([32, 186, 286]), nn.MaxPool2d(2), nn.ReLU(), 
      nn.Conv2d(32, 64, 5 ), nn.ReLU(), nn.Conv2d(64, 128, 3),
      nn.LayerNorm([128, 87, 137]), nn.MaxPool2d(2), nn.ReLU(), 
      nn.Conv2d(128, 256, 3), nn.ReLU(), nn.Conv2d(256, 128, 3),
      nn.LayerNorm([128, 39, 64]), nn.MaxPool2d(2), nn.ReLU(), nn.Flatten(),
      nn.Linear(128*19*32, 2**13),
    )
  
  def forward(self, x):
    x = x.unsqueeze(1)
    return self.blocks(x)


model = Model().to('cuda')
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
p = model(x)
# %%

def train_step(x, y, label):
  embx = model(x)
  emby = model(y)

  targets = torch.where(label>0, torch.tensor([1.]).to('cuda'), torch.tensor([-1.]).to('cuda'))
  loss = nn.functional.cosine_embedding_loss(embx, emby, targets)
  opt.zero_grad()
  loss.backward()
  opt.step()
  return loss

#%%

def test():
  with torch.no_grad():
    x,y,labels = getsample(10)
    embx = model(x)
    emby = model(y)
    targets = torch.where(labels>0, torch.tensor([1.]).to('cuda'), torch.tensor([-1.]).to('cuda'))

    similarity = nn.functional.cosine_similarity(embx, emby)
    acc = ((similarity * targets) > 0).float().mean()
    loss = nn.functional.cosine_embedding_loss(embx, emby, targets)
    print (f'acc: {acc.item():6.2f} loss: {loss.item():6.2f}')

#%%

for i in range(1000):
  x,y,labels = getsample(10)
  loss = train_step(x, y, labels)
  print(f'\r loss: {loss.item():6.2f} ', end='')
  if not i or (i-1) % 10 == 0: test()

# %%
