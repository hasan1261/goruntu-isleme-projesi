import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Karakalem Efekti", layout="wide")
st.title("✏️ Karakalem Efekti ")

# Fotoğraf yükleme
uploaded_file = st.file_uploader("Bir fotoğraf seçin", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # PIL -> OpenCV
    image = Image.open(uploaded_file).convert('RGB')
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    # --- Karakalem Efekti ---
    
    # 1. Gri ton
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    # 2. CLAHE ile kontrast artır
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # 3. Dodge blend
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (11, 11), 0)
    sketch = cv2.divide(gray, 255 - blurred, scale=230)

    # 4. Göz ve detayları korumak için çok hafif keskinleştirme
    kernel = np.array([[0, -0.5, 0],
                       [-0.5, 3, -0.5],
                       [0, -0.5, 0]])
    sketch = cv2.filter2D(sketch, -1, kernel)

    # 5. Tek kanal -> renkli format (saç rengi fark etmez)
    sketch_final = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    # PIL formatına çevir
    sketch_pil = Image.fromarray(cv2.cvtColor(sketch_final, cv2.COLOR_BGR2RGB))

    # Orijinal ve karakalem yan yana
    st.write("Orijinal ve Karakalem Fotoğraf:")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Orijinal Fotoğraf", use_container_width=True)
    with col2:
        st.image(sketch_pil, caption="Karakalem Efekti", use_container_width=True)