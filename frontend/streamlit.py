import io

import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder

import streamlit as st

backend = "http://client:5000/upload/image"


def process(image, server_url: str):

    m = MultipartEncoder(fields={"file": ("filename", image, "image/jpeg")})

    r = requests.post(
        server_url, data=m, headers={"Content-Type": m.content_type}, timeout=8000
    )

    return r


st.title("Text Recognition using Triton Inference Server and FastAPI Demo")
st.divider()

st.write(
    """Simple OCR process to extract text from images using Triton server to serve the models and FastAPI to create the API
    . The demo comes with grafana and prometheus to monitor the server. 
    All credit on the Triton code goes to https://github.com/triton-inference-server/tutorials. This is not ready for production,
    it is just a demo to show how to use Triton with FastAPI.
    """
)
col1, col2 = st.columns(2)
with col1:
    st.link_button("Go to grafana", "http://localhost:4669", use_container_width=True, type='primary')
with col2:
    st.link_button("Go to prometheus", "http://localhost:9090", use_container_width=True)

st.divider()
st.subheader("Upload an image")
input_image = st.file_uploader("insert image")

# show preview
if input_image:
    try:
        original_image = Image.open(input_image).convert("RGB")
        st.image(original_image, use_column_width='auto')
    except:
        st.write("File is not an image!")

if st.button("Extract text!"):

    col1, col2 = st.columns(2)

    if input_image:
        with st.spinner('Wait for it...'):
            try:
                text_array = process(input_image, backend)
                st.subheader("Text extracted in no particular order :white_check_mark:")
                st.write(text_array.json()["text"])
            except:
                st.write("File is not an image!")
    else:
        # handle case with no image
        st.write("Insert an image!")