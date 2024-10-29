import os

import torch
import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from data import *
from net import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = 'params/unet.pth'
data_path = r'data'
save_path = 'train_image'
if __name__ == '__main__':
    data_loader = DataLoader(MyDataset(data_path), batch_size=1, shuffle=True)

    net = UNet().to(device)  ##将网络模型移到GPU上
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight！')
    else:
        print('not successful load weight')

    opt = optim.Adam(net.parameters())
    loss_fun = nn.BCELoss()

    epoch = 1
    while epoch < 200:
        for i, (image, segment_image) in enumerate(data_loader):
            image, segment_image = image.to(device), segment_image.to(device)   ##将数据移到GPU上
            out_image = net(image)
            train_loss = loss_fun(out_image, segment_image)
            opt.zero_grad()
            train_loss.backward()
            opt.step()

            if i % 1 == 0:
                print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')

            _image = image[0]
            _segment_image = segment_image[0] 
            _out_image = out_image[0] * 255

            img = torch.stack([_segment_image, _out_image], dim=0)
            save_image(img, f'{save_path}/{i}.png')
        if epoch % 5 == 0:
            torch.save(net.state_dict(), weight_path)
            print('save successfully!')
        epoch += 1
