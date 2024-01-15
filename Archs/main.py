import torch
import torch.nn as nn
from NafnetArch import NAFNet

device = torch.device("cuda")

img_channel = 3
width = 32

enc_blks = [1, 1, 1, 28]
middle_blk_num = 1
dec_blks = [1, 1, 1, 1]

model = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
               enc_blk_nums=enc_blks, dec_blk_nums=dec_blks).to(device)

criterion = nn.MSELoss()
lr = 0.0001
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
