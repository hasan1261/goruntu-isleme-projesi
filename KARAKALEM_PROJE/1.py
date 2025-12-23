import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Karakalem Efekti", layout="wide")
st.title(" Karakalem Efekti ")

uploaded_file = st.file_uploader("Bir fotoğraf seçin", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (11, 11), 0)
    sketch = cv2.divide(gray, 255 - blurred, scale=230)

    kernel = np.array([[0, -0.5, 0],
                       [-0.5, 3, -0.5],
                       [0, -0.5, 0]])
    sketch = cv2.filter2D(sketch, -1, kernel)

    sketch_final = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    sketch_pil = Image.fromarray(cv2.cvtColor(sketch_final, cv2.COLOR_BGR2RGB))

    st.write("Orijinal ve Karakalem Fotoğraf:")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Orijinal Fotoğraf", use_container_width=True)
    with col2:

        st.image(sketch_pil, caption="Karakalem Efekti", use_container_width=True)
