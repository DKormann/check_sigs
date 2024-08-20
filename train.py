#%%
import os 
from tinygrad import Tensor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
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
  for img_path in os.listdir(path):
    if not img_path.endswith('.png'): continue
    id = img_path.split('_')[1]
    collection.setdefault(id, []).append(loadimage(path+'/'+img_path))
  return np.array([np.array(collection[key]) for key in sorted(collection.keys())])

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
  s = np.random.randint(24, size=n)
  arr = np.where(labels, train_org[idxs, s], train_forg[idxs, s])
  print(arr.shape)

getsample(40)

#%%


arr = np.random.rand(10,10,10)

ia = np.random.randint(10, size=(3,))
ib = np.random.randint(10, size=(3,))

arr[ia, ib].shape

