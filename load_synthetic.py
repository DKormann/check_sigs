#%%
from PIL import Image
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from typing import Tuple

datapath = '/shared/datasets/signatures/synthetic_gpds_preprocessed_v2'

os.listdir(datapath)

def random_pair(datapath = datapath, ds_offset=0, ds_size = 10_000, b_size = 24):
  n = np.random.randint(ds_offset, ds_size)+1
  label = np.random.randint(2)
  c = np.random.randint(b_size)+1
  f = np.random.randint(b_size)+1
  cp = datapath + f"/genuine/{n:03}/c-{n:03}-{c:0>2}.jpg"
  if label: fp = datapath + f"/genuine/{n:03}/c-{n:03}-{f:0>2}.jpg"
  else: fp = datapath + f"/forged/{n:03}/cf-{n:03}-{f:0>2}.jpg"
  return cp, fp, label

#%%

def loadimage(path:str): return np.array(Image.open(path).convert('L'))

def sample(n=40, datapath = datapath,ds_offset=0, ds_size= 10_000, b_size= 24)->Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """returns np.arrays of Real, [Fake | Real], Labels"""
  r,f,l = zip(*[random_pair(datapath, ds_offset=ds_offset, ds_size = ds_size, b_size=b_size) for _ in range(n)])
  r = np.stack([loadimage(p) for p in r])/255.0
  f = np.stack([loadimage(p) for p in f])/255.0
  l = np.array(l)
  return r, f, l
#%%

# non random sampling for test
def test_sample(datapath = datapath, ds_size = 20, b_size = 24):
  n = np.concatenate([np.arange(ds_size*2)]) % ds_size +1
  label = np.concatenate([np.zeros(ds_size), np.ones(ds_size)])
  r,f,labs = [], [], []
  img_per_set = 16
  for n, l in zip(n, label):
    for i in range(1,img_per_set+1):
      i_ = (i+1) % img_per_set + 1
      r.append(loadimage(datapath + f"/genuine/{n:03}/c-{n:03}-{i:0>2}.jpg"))
      f.append(loadimage(datapath + (f"/genuine/{n:03}/c-{n:03}-{i_:0>2}.jpg" if l else f"/forged/{n:03}/cf-{n:03}-{i:0>2}.jpg")))
      labs.append(l)
  return np.stack(r)/255.0, np.stack(f)/255.0, np.array(labs)

  