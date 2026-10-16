import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass, shift

st.title("Rozpoznawanie cyfr przez AI")
st.write("Narysuj cyfrę na środku:")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=30,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
    return_image_data=True,
)

if st.button("Sprawdź co narysowałem"):
    if canvas_result.image_data is not None:
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

        st.image(Image.fromarray((img_array * 255).astype("uint8")).resize((140, 140)), caption="Tak to widzi model (28x28)")

        img_vector = img_array.reshape(1, 784)
        
        try:
            data = np.load('wagi_modelu.npz')
            W1, b1, W2, b2 = data['W1'], data['b1'], data['W2'], data['b2']
            
            z1 = np.dot(img_vector, W1) + b1
            a1 = np.maximum(0, z1)
            z2 = np.dot(a1, W2) + b2
            
            exp = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
            probs = exp / np.sum(exp, axis=1, keepdims=True)
            
            cyfra = np.argmax(probs)
            pewnosc = np.max(probs) * 100
            
            st.success(f"### Model myśli, że to cyfra: {cyfra} (Pewność: {pewnosc:.1f}%)")
            st.bar_chart(probs[0])
            
        except FileNotFoundError:
            st.error("Najpierw uruchom plik `model.py`!")