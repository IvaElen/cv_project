import streamlit as st
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import make_grid
import numpy as np
from random import randrange



st.title('Пишем числа, как врачи :black_nib: :pill:')

path = 'CondGAN_dict.pth'

def load_model(path):

    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Sequential(
                nn.Linear(110, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 784),
                nn.Sigmoid()
            )

        def forward(self, x, labels):
            out = self.layer1(torch.cat((x, labels), 1))
            return out.view(out.size(0), 1, 28, 28)
        
        
    model = Generator()
    model.load_state_dict(torch.load('CondGAN_dict.pth',  map_location=torch.device('cpu')))

    return model

model = load_model(path)
model.eval()


digit_list = st.text_input('**Введите число**', '42')


def digit_generator(digit):    
    
    latent = torch.randn(1, 100)
    target = F.one_hot(torch.tensor(digit), 10)
    target = target.view(1, target.size(0))

    result = model(latent, target)[:, 0, :, :].permute(1, 2, 0).detach().numpy()

    clarity_image = np.zeros_like(result)

    for ind_row, row in enumerate(result):
        new_row = np.zeros(row.shape)
        for ind_col, col in enumerate(row):
            if col<0.5:
                new_row[ind_col] = 1.
            elif col>0.5:
                new_row[ind_col] = 0.
            else:
                new_row[ind_col]= row[ind_col]
        clarity_image[ind_row] = new_row
    
    return clarity_image

def digit_output(digit_list):   
    
    img_list = []

    for digit in digit_list:
        img_list.append(digit_generator(int(digit)))

    if len(img_list) <= 7:
        w = 100
    elif 8 <= len(img_list) <= 11:
        w = 60
    else:
        w = 30
     
    return img_list, w

images, w = digit_output(digit_list)

st.image(images, width=w)

path2 = 'punks.pth'

def load_model2(path):
    
    class Generator(nn.Module):
        def __init__(self):
            super().__init__()

            self.layer1 = nn.Sequential(
                nn.ConvTranspose2d(64, 512, kernel_size=4, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(inplace=True)
            )

            self.layer2 = nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(inplace=True)
            )

            self.layer3 = nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(inplace=True)
            )

            self.layer4 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(inplace=True)
            )

            self.layer5 = nn.Sequential(
                nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
                nn.Tanh()
            )
        
        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.layer5(out)

            return out
    
    model = Generator()
    model.load_state_dict(torch.load('punks.pth',  map_location=torch.device('cpu')))

    return model

model2 = load_model2(path)
model2.eval()

def denorm(img_tensors):
        stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        return img_tensors * stats[1][0] + stats[0][0]

def punks_generate(num):
    
    punks = []
    for i in range(1, num + 1):
        result = torch.squeeze(model2(torch.randn(1, 64, 1, 1))).detach().permute(1, 2, 0).numpy()
        punks.append(denorm(result))

    return punks

st.divider()

st.title('Давай глянем скольким панкам нужно к врачу 🤘')

if st.button('Панки'):
    st.image(punks_generate(randrange(1, 20)), width=100)
else:
    st.write('**Пока их 0!**')
