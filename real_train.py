#%%
import torch, os
from model import Model
import  load_synthetic
import sys

datapath = '/shared/datasets/signatures/gpds150_preprocessed/train'

device = 'cuda:1'
model = Model().to(device)



if sys.argv[-1]=="--pretrained" and  os.path.exists('model_pretrained_e_10000.pth'):
  model.load_state_dict(torch.load('model_pretrained_e_10000.pth'))
  print('Model loaded')

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
#%%
def sample():
  r,f,l = load_synthetic.sample(40, datapath,ds_offset=20, ds_size=150, b_size=16)
  r = torch.tensor(r, dtype=torch.float32).to(device)
  f = torch.tensor(f, dtype=torch.float32).to(device)
  l = torch.tensor(l, dtype=torch.bool).to(device)
  return r,f,l

sample()

#%%
def trainstep(x, y, labels):
  model.train()
  opt.zero_grad()
  xembed = model(x)
  yembed = model(y)
  targets = torch.where(labels, torch.ones(labels.shape).to(device), -torch.ones(labels.shape).to(device))
  loss = torch.nn.functional.cosine_embedding_loss(xembed, yembed, targets)
  loss.backward()
  opt.step()
  return loss.detach().cpu().numpy()

testr, testf, testl = load_synthetic.test_sample(datapath, ds_size=20, b_size=16)
testr = torch.tensor(testr, dtype=torch.float32).to(device)
testf = torch.tensor(testf, dtype=torch.float32).to(device)
testl = torch.tensor(testl, dtype=torch.bool).to(device)

import numpy as np

def test():
  with torch.no_grad():
    embx = model(testr)
    emby = model(testf)
    targets = torch.where(testl, torch.ones(testl.shape).to(device), -torch.ones(testl.shape).to(device))
    loss = torch.nn.functional.cosine_embedding_loss(embx, emby, targets).detach().cpu().numpy()
    sims = torch.nn.functional.cosine_similarity(embx, emby).detach().cpu().numpy()
    genuine_sims = sims[testl.cpu()]
    forge_sims = sims[~testl.cpu()]

    d = 0.7
    acc = np.concatenate([genuine_sims>d, forge_sims<=d]).mean()
    return f' test loss: {loss:6.2f}  best d:{d:5.2f} accuracy: {acc*100:6.2f}%'


#%%
if __name__ == '__main__':
  EPOCHS = 2000

  loss_avg = 0
  for e in range(EPOCHS):
    r,f,l = sample()
    loss = trainstep(r, f, l)
    loss_avg =  0.9*loss_avg + 0.1*loss if loss_avg else loss
    
    print(f'\repchs:{e} train loss:{loss_avg:6.4f}', end='')
    if e == 0 or (e+1) % 100 == 0:
      print(' ', (testval:=test()))
      torch.save(model.state_dict(), f'model_finetuned_e_{e+1}_{testval}.pth')

