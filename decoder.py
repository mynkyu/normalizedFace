#F to 12 x 12 x 256
#stacked transposed convolutions, separated by ReLUs, with a kernel width of 5 and stride of 2 to upsample to 96 x 96 x 32
#1 x 1 convolution to 96 x 96 x 3

# torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

f_vector = 736 # nn4 model의 output layer 전 마지막 layer dimension

class Decoder(nn.Module):
    
    def __init__(self, useCude=False):
        super(Decoder, self).__init__()
        
        self.fc = nn.Linear(f_vector, 256 * 12 * 12)
        
        self.tConv1 = nn.ConvTranspose2d(256, 128, 5, stride=2, padding=2)
        self.tConv2 = nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2)
        self.tConv3 = nn.ConvTranspose2d(64, 32, 5, stride=2, padding=2)
        
        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU())
        
    def forward(self, x):
        out = self.fc(x)
        out = out.view(1, 256, 12, 12)
        out = F.relu(self.tConv1(out, output_size=[None, None, 24, 24]))
        out = F.relu(self.tConv2(out, output_size=[None, None, 48, 48]))
        out = F.relu(self.tConv3(out, output_size=[None, None, 96, 96]))
        out = self.conv(out)
        return out

dtype = torch.FloatTensor

# input : 736 dimension
x = Variable(torch.randn(1, 1, 1, f_vector))

# ground_truth_textures := actual input image
#y = Variable(ground_truth_textures, requires_grad=False)
y = Variable(torch.randn(1, 3, 96, 96), requires_grad=False)

net = Decoder()

loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-4
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

for t in range(500):
    y_pred = net(x) # [1, 3, 96, 96]

    loss = loss_fn(y_pred, y)
    if (t % 100 == 0):
        print(t, loss.data[0])

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
