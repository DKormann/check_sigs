
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
