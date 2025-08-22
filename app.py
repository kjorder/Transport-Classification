import streamlit as st
from pathlib import Path
from fastai.learner import load_learner
from PIL import Image

@st.cache_resource
def load_model():
    p = Path(__file__).parent / "modelll.pkl"   # yoki modelll_new.pkl
    learn = load_learner(p)
    return learn

st.title("Image Classifier")

uploaded = st.file_uploader("Rasm yuklang", type=["jpg","jpeg","png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Yuklangan rasm", use_column_width=True)
    learn = load_model()
    pred, pred_idx, probs = learn.predict(img)
    st.success(f"Natija: {pred} ({float(probs[pred_idx])*100:.1f}%)")


