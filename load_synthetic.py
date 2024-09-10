#%%
from PIL import Image
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from typing import Tuple

datapath = '/shared/datasets/signatures/synthetic_gpds_preprocessed'

os.listdir(datapath)

def random_pair():
  n = np.random.randint(10_000)+1
  label = np.random.randint(2)
  c = np.random.randint(24)+1
  f = np.random.randint(24)+1
  cp = datapath + f"/genuine/{n:03}/c-{n:03}-{c:0>2}.jpg"
  if label: fp = datapath + f"/genuine/{n:03}/c-{n:03}-{f:0>2}.jpg"
  else: fp = datapath + f"/forged/{n:03}/cf-{n:03}-{f:0>2}.jpg"
  return cp, fp, label

random_pair()


#%%

def loadimage(path:str):
  img = Image.open(path).convert('L')
  return np.array(img)

def sample(n=40)->Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """returns np.arrays of Real, [Fake | Real], Labels"""
  r,f,l = zip(*[random_pair() for _ in range(n)])
  r = np.stack([loadimage(p) for p in r])
  f = np.stack([loadimage(p) for p in f])
  l = np.array(l)
  return r, f, l
