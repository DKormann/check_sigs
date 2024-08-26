#%%
import os 
from tinygrad import Tensor, dtypes, TinyJit
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tinygrad.helpers import trange, tqdm
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
  t = tqdm(os.listdir(path), desc=f'loading {path}')
  for img_path in t:
    if not img_path.endswith('.png'): continue
    id = img_path.split('_')[1]
    collection.setdefault(id, []).append(loadimage(path+'/'+img_path))
  return np.array([np.array(collection[key]) for key in sorted(collection.keys())], dtype=np.float16) / 255.0

org = readpath(datapath+'/full_org')
forg = readpath(datapath+'/full_forg')

train_org = org[:50]
train_forg = forg[:50]
test_org = org[50:]
test_forg = forg[50:]

def getsample(n=50):
  labels = np.random.randint( 2, size=(n,1,1))
  idxs = np.random.randint(50, size=n)
  xn = np.random.randint(24, size=n)
  x = train_org[idxs, xn]
  yn = np.random.randint(24, size=n)
  y = np.where(labels, train_org[idxs, yn], train_forg[idxs, yn])
  return Tensor(x).realize(), Tensor(y).realize(), Tensor(labels.flatten(), dtype=dtypes.int32).realize()

x,y,labels = getsample(10)

x.shape, y.shape, labels.shape

#%%
from tinygrad import nn
from tinygrad.nn import Conv2d

class Norm():
  def __init__(self, dim=32): self.norm = nn.LayerNorm(dim)
  def __call__(self, x:Tensor): return self.norm(x.transpose(-3,-1)).transpose(-3,-1)

class Model():
  def __init__(self):
    self.layers =[
      lambda x: x.unsqueeze(1),
      Conv2d(1, 96, 11, 5), Tensor.gelu,
      Conv2d(96, 128, 5, 2), Tensor.gelu,
      Norm(128), Tensor.max_pool2d,
      Conv2d(128, 256, 3), Tensor.gelu,
      Conv2d(256, 128, 3), Tensor.gelu,
      Norm(128), Tensor.max_pool2d,
      Conv2d(128, 128, 3), Tensor.gelu,
      Norm(128), Tensor.max_pool2d,
      lambda x: x.flatten(1),
    ]

    rawdim = 6144 // 2

    self.join = [
      nn.Linear(rawdim * 2, rawdim), Tensor.gelu,
      nn.Linear(rawdim, 1), Tensor.sigmoid
    ]
  
  def __call__(self, x ,y):
    x = x.sequential(self.layers)
    y = y.sequential(self.layers)
    z = Tensor.cat(x,y, dim=1)
    return z.sequential(self.join)

model = Model()
opt = nn.optim.Adam(nn.state.get_parameters(model.layers))



@TinyJit
def train_step(x,y,labels):
  with Tensor.train():
    opt.zero_grad()
    Tensor.training = True
    pred = model(x,y)
    loss = pred.binary_crossentropy(labels).backward()
    opt.step()
    return loss.realize()

for i in (t:=trange(1000)):
  x,y,labels = getsample(64)
  loss = train_step(x,y,labels)
  t.set_description(f'loss: {loss.item():6.2f}')

#%%
train_step(x,y, labels).numpy()


