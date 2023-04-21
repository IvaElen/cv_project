import glob
import streamlit as st
import wget
from PIL import Image
import torch
import os
import time

st.set_page_config(layout="wide")

# cfg_model_path = 'models/last.pt'

confidence = .25


def image_input(data_src):
    img_file = None
    img_bytes = st.sidebar.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])
    if img_bytes:
        img_file = "pages/data/uploaded_data/upload." + img_bytes.name.split('.')[-1]
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
def load_model(device):
    model = torch.load('models/yolov5s.pt')
    model.load_state_dict(torch.load('models/last.pt'))
    model.to(device)
    print("model to ", device)
    return model


# def get_user_model():
#     model_src = st.sidebar.radio("Model source", ["file upload", "url"])
#     model_file = None
#     if model_src == "file upload":
#         model_bytes = st.sidebar.file_uploader("Upload a model file", type=['pt'])
#         if model_bytes:
#             model_file = "models/uploaded_" + model_bytes.name
#             with open(model_file, 'wb') as out:
#                 out.write(model_bytes.read())
#     else:
#         url = st.sidebar.text_input("model url")
#         if url:
#             model_file_ = download_model(url)
#             if model_file_.split(".")[-1] == "pt":
#                 model_file = model_file_

#     return model_file

def main():
    # global variables
    global model, confidence, cfg_model_path

    st.title("Object Recognition YOLO V5")

#     st.sidebar.title("Settings")

    # upload model
    model_src = "Use our demo model 5s"

    # check if model file is available
    if not os.path.isfile(cfg_model_path):
        st.warning("Model file not available!!! Please added to the model folder.", icon="⚠️")
    else:

        device_option = 'cpu'
        # load model
        model = load_model(cfg_model_path, device_option)

        # confidence slider
        confidence = st.sidebar.slider('Confidence', min_value=0.1, max_value=1.0, value=.45)


        st.sidebar.markdown("---")

        # input options
#         input_option = st.sidebar.radio("Select input type: ", ['image', 'video'])

        # input src option
        data_src = 'Upload your own data'
        input_option = 'image'
        image_input(data_src)
#         else:
#             video_input(data_src)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass

