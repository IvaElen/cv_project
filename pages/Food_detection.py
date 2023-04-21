import glob
import streamlit as st
import wget
from PIL import Image
import torch
import cv2
import os
import time

st.set_page_config(layout="wide")

cfg_model_path = 'models/last.pt'
model = None

confidence = .25


def image_input(data_src):
    img_file = None
    img_bytes = st.sidebar.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])
    if img_bytes:
        img_file = "data/uploaded_data/upload." + img_bytes.name.split('.')[-1]
        Image.open(img_bytes).save(img_file)

    if img_file:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_file, caption="Selected Image")
        with col2:
            img = infer_image(img_file)
            st.image(img, caption="Model prediction")


def infer_image(img, size=None):
    model.conf = confidence
    result = model(img, size=size) if size else model(img)
    result.render()
    image = Image.fromarray(result.ims[0])
    return image


@st.cache_resource
def load_model(path, device):
    model_ = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)
    model_.to(device)
    print("model to ", device)
    return model_


#@st.experimental_singleton
@st.cache_resource
def download_model(url):
    model_file = wget.download(url, out="models")
    return model_file


def main():
    # global variables
    global model, confidence, cfg_model_path

    st.title("Object Recognition YOLO V5")

    st.sidebar.title("Settings")

    # upload model
    model_src = "Use our demo model 5s"

    # check if model file is available
    if not os.path.isfile(cfg_model_path):
        st.warning("Model file not available!!!, please added to the model folder.", icon="⚠️")
    else:

        device_option = 'cpu'
        # load model
        model = load_model(cfg_model_path, device_option)

        # confidence slider
        confidence = st.sidebar.slider('Confidence', min_value=0.1, max_value=1.0, value=.45)


        st.sidebar.markdown("---")

        # input options
        #input_option = st.sidebar.radio("Select input type: ", ['image', 'video'])
        input_option ='image'
        # input src option
        data_src = 'Upload your own data'
        if input_option == 'image':
            image_input(data_src)

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
