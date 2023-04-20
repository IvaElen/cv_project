import io 
import numpy as np
import streamlit as st
import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image

def load_model():

    class ConvAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()

            # Часть экодера
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3),
                nn.ReLU(),

                nn.Conv2d(32, 64, kernel_size=3),
                nn.BatchNorm2d(64),
                nn.Dropout(),
                nn.ReLU()
                )
            
            self.pool = nn.MaxPool2d(2, 2, return_indices=True, ceil_mode=True) #<<<<<< Bottleneck
            
            # часть декодера
            self.unpool = nn.MaxUnpool2d(2, 2)

            self.decoder = nn.Sequential(

                nn.ConvTranspose2d(64, 32, kernel_size=3),
                nn.BatchNorm2d(32),
                nn.ReLU(),

                nn.ConvTranspose2d(32, 1, kernel_size=3),
                nn.Sigmoid())
            
        def encode(self, x):
            x = self.encoder(x)
            x, indicies = self.pool(x) # ⟸ bottleneck
            return x, indicies

        def decode(self, x, indicies):
            x = self.unpool(x, indicies)
            x = self.decoder(x)
            return x

        def forward(self, x):
            latent, indicies = self.encode(x)
            out = self.decode(latent, indicies)      
            return out

    model = ConvAutoencoder()
    model.load_state_dict(torch.load('model_weights.pth'))
    return model

def load_image():
    uploaded_file = st.file_uploader(label = 'Загрузите изображение, которое нужно очистить от шумов')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None

def preprocess_image(img):
    preprocess = T.Compose([
            T.ToTensor()])
    st.write(f'**Размер картинки: {preprocess(img).shape[1]} x {preprocess(img).shape[1]}**')
    st.write(f'**Количество каналов: {preprocess(img).shape[0]}**')
    image_for_model = preprocess(img)[-1::].unsqueeze(0)
    return image_for_model/255

def clean_image(image_for_model):
    predict = model(image_for_model.unsqueeze(0)[0])
    return predict

model = load_model()

st.title('Очистка сканов документов от шумов')
img = load_image()
black = st.slider('Интенсивность черного', min_value=0., max_value=0.5, value=0.)
white = st.slider('Интенсивность белого', min_value=0.5, max_value=1., value=1.)
result = st.button('Очистить изображение!')
if result:
    prepros = preprocess_image(img)
    clear_img = clean_image(prepros)
    clarity_image = np.zeros_like(clear_img[0][0].detach().numpy())
    for ind_row, row in enumerate(clear_img[0][0]):
        new_row = np.zeros(row.shape)
        for ind_col, col in enumerate(row):
            if col>white:
                new_row[ind_col] = 1.0
            elif col<black:
                new_row[ind_col] = 0.
            else:
                new_row[ind_col]= row[ind_col]
        clarity_image[ind_row] = new_row
    st.image(clarity_image)
    