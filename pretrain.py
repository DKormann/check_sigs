#%%
from model import Model
from load_synthetic import sample


#%%

model = Model()


#%%
import torch 
import os

device = 'cuda:1'

model = Model().to(device)

if os.path.exists('model.pth'):
  model.load_state_dict(torch.load('model.pth'))
  print('Model loaded')

#%%

r,f,l = sample(40)

r = torch.tensor(r, dtype=torch.float32).to(device)
f = torch.tensor(f, dtype=torch.float32).to(device)
l = torch.tensor(l, dtype=torch.bool).to(device)

#%%

r.shape

#%%

model.train(False)

model(r)

