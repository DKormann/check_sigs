#%%
import torch, os
from model import Model
import  load_synthetic


datapath = '/shared/datasets/signatures/gpds150_preprocessed/train'

device = 'cuda:1'
model = Model().to(device)


if os.path.exists('model_pretrained_e_10000.pth'):
  model.load_state_dict(torch.load('model_pretrained_e_10000.pth'))
  print('Model loaded')

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
#%%
def sample(test=False):
  r,f,l =  load_synthetic.sample(40, datapath, ds_size=20, b_size=16) if test else load_synthetic.sample(40, datapath,ds_offset=20, ds_size=150, b_size=16)
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





# def test():
#   with torch.no_grad():
#     x,y,labels = getsample(10)
#     embx = model(x)
#     emby = model(y)
#     targets = torch.where(labels>0, torch.tensor([1.]).to('cuda'), torch.tensor([-1.]).to('cuda'))

#     similarity = nn.functional.cosine_similarity(embx, emby)
#     acc = ((similarity * targets) > 0).float().mean()
#     loss = nn.functional.cosine_embedding_loss(embx, emby, targets)
#     print (f'acc: {acc.item():6.2f} loss: {loss.item():6.2f}')




#%%

EPOCHS = 1000

loss_avg = 0
for e in range(EPOCHS):
  r,f,l = sample()
  loss = trainstep(r, f, l)
  loss_avg =  0.9*loss_avg + 0.1*loss if loss_avg else loss
  
  print(f'\repchs:{e} train loss:{loss_avg:6.4f}', end='')
  if e == 0 or (e+1) % 100 == 0: print()


torch.save(model.state_dict(), f'model_finetuned_e_{e+1}.pth')