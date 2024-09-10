#%%
import torch, os
from model import Model
import  load_synthetic

device = 'cuda:1'
model = Model().to(device)

if os.path.exists('model.pth'):
  model.load_state_dict(torch.load('model.pth'))
  print('Model loaded')

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
#%%
def sample():
  r,f,l = load_synthetic.sample(40)

  r = torch.tensor(r, dtype=torch.float32).to(device)
  f = torch.tensor(f, dtype=torch.float32).to(device)
  l = torch.tensor(l, dtype=torch.bool).to(device)
  return r,f,l

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


#%%

EPOCHS = 10_000

loss_avg = 0
for e in range(EPOCHS):
  r,f,l = sample()
  loss = trainstep(r, f, l)
  loss_avg = 0.9*loss_avg + 0.1*loss
  
  print(f'\r{e}: {loss_avg:6.4f}', end='')
  if e == 0 or (e+1) % 100 == 0: print()
