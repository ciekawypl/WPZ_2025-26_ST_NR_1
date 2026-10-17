import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass, shift

st.set_page_config(layout="wide", page_title="Rozpoznawanie ręcznie pisanych cyfr")

st.markdown("""
<style>
div.stButton > button {
    width: 280px !important;
    display: block;
    margin: 0 auto;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; margin-bottom: 10px;'>Rozpoznawanie ręcznie pisanych cyfr</h2>", unsafe_allow_html=True)

if 'canvas_key' not in st.session_state:
    st.session_state.canvas_key = 0

_, center_col, _ = st.columns([2, 5, 2])

with center_col:
    col1, col2 = st.columns([1, 1.1], gap="medium")

    with col1:
        st.markdown("#### 1. Narysuj cyfrę")
        
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=30,
            stroke_color="white",
            background_color="black",
            width=280,
            height=280,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.canvas_key}",
            return_image_data=True,
        )
        
        if st.button("Wyczyść rysunek"):
            st.session_state.canvas_key += 1
            st.rerun()

    with col2:
        st.markdown("#### 2. Wynik modelu")
        
        if canvas_result.image_data is not None and canvas_result.image_data.sum() > 0:
            img_data = canvas_result.image_data.astype("uint8")
            img = Image.fromarray(img_data).convert("L")
            img = img.resize((28, 28), Image.Resampling.LANCZOS)
            img_array = np.array(img).astype("float32") / 255.0
            img_array = np.where(img_array > 0.15, img_array, 0.0)

            if img_array.sum() > 0:
                cy, cx = center_of_mass(img_array)
                rows, cols = img_array.shape
                shiftx = np.round(cols / 2.0 - cx).astype(int)
                shifty = np.round(rows / 2.0 - cy).astype(int)
                img_array = shift(img_array, [shifty, shiftx], cval=0)

            img_vector = img_array.reshape(1, 784)
            
            try:
                data = np.load('model.npz')
                W1, b1, W2, b2 = data['W1'], data['b1'], data['W2'], data['b2']
                
                z1 = np.dot(img_vector, W1) + b1
                a1 = np.maximum(0, z1)
                z2 = np.dot(a1, W2) + b2
                
                exp = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
                probs = exp / np.sum(exp, axis=1, keepdims=True)
                
                cyfra = np.argmax(probs)
                pewnosc = np.max(probs) * 100
                
                st.success(f"**Rozpoznana cyfra: {cyfra}**  \nPewność: **{pewnosc:.1f}%**")
                st.bar_chart(probs[0], height=180)
                
            except FileNotFoundError:
                st.error("Nie znaleziono modelu!")
        else:
            st.info("Narysuj cyfrę na planszy.")