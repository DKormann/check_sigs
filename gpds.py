
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

def getsample(n=50)->torch.Tensor:
  labels = torch.randint(2, (n, 1, 1)).type(torch.bool)
  idxs = torch.randint(120, (n,))
  xn = torch.randint(16, (n,))
  x = train_genuine[idxs, xn].to(device=device, dtype=torch.float32)
  yn = torch.randint(16, (n,))
  y = torch.where(labels, train_genuine[idxs, yn], train_forge[idxs, yn]).to(device=device, dtype=torch.float32)
  return x, y, labels.flatten().to(device)

x,y,labels = getsample(10)

#%%

testx = torch.cat([test_genuine[:,0], test_genuine[:,1]], 0).to(device, torch.float32)
testy = torch.cat([test_genuine[:,2], test_forge[:,0]], 0).to(device, torch.float32)
testlabels = torch.cat([torch.ones(30), torch.zeros(30)], 0).to(device, torch.bool)

#%%
from torch import nn

class Model(nn.Module):

  def __init__(self):
    super(Model, self).__init__()
    self.blocks = nn.Sequential(
      nn.Conv2d(1, 16, 11), nn.ReLU(), nn.Conv2d(16, 32, 5 ),
      nn.LayerNorm([32, 186, 286]), nn.MaxPool2d(2), nn.ReLU(), 
      nn.Dropout(0.3),
      nn.Conv2d(32, 64, 5 ), nn.ReLU(), nn.Conv2d(64, 128, 3),
      nn.LayerNorm([128, 87, 137]), nn.MaxPool2d(2), nn.ReLU(), 
      nn.Dropout(0.3),
      nn.Conv2d(128, 256, 3), nn.ReLU(), nn.Conv2d(256, 128, 3),
      nn.LayerNorm([128, 39, 64]), nn.MaxPool2d(2), nn.ReLU(), nn.Flatten(),
      nn.Linear(128*19*32, 2**13),
      nn.ReLU(), nn.Linear(2**13, 2**7),
    )
  
  def forward(self, x): return self.blocks(x.unsqueeze(1))


EPOCHS=1000
torch.manual_seed(0)
model = Model().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)


scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=EPOCHS)
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


def test(plot = False):
  model.eval()
  with torch.no_grad():

    xembed = model(testx)
    yembed = model(testy)
    
    targets = torch.where(testlabels, torch.ones(testlabels.shape).to(device), -torch.ones(testlabels.shape).to(device))
    loss = torch.nn.functional.cosine_embedding_loss(xembed, yembed, targets).detach().cpu().numpy()

    sims = torch.nn.functional.cosine_similarity(xembed, yembed).detach().cpu().numpy()

    genuine_sims = sims[:30]
    forge_sims = sims[30:]

    if plot:
      plt.hist(genuine_sims, bins=20, alpha=0.5, label='genuine')
      plt.hist(forge_sims, bins=20, alpha=0.5, label='forge')

    acc, d = max((np.concatenate([genuine_sims>d, forge_sims<=d]).mean(),d) for d in np.linspace(0, 1, 101))
    return f' loss: {loss:6.2f}  best d:{d:5.2f} accuracy: {acc*100:6.2f}%'

print(test(plot=True))

#%%
bs = 20

for i in range(EPOCHS):
  x,y,labels = getsample(bs)
  loss = trainstep(x, y, labels)
  print(f'\r{i}: {loss:6.2f}', end='')
  if not i or (i+1) % 10 == 0: print(test())

test(True)

#%%
for i in range(20):
  k = i

  g = test_genuine[k:k+1,0].to(dtype=torch.float32, device = device)
  f= test_genuine[k:k+1,2].to(dtype=torch.float32, device = device)

  genuine_embed = model(g)
  forge_embed = model(f)

  similarity = torch.nn.functional.cosine_similarity(genuine_embed, forge_embed)

  print(similarity)
