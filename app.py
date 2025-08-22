import streamlit as st
from fastai.vision.all import *
# After
pip install fasttransform
from fasttransform import Transform, Pipeline
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# title
st.title('Mashina, Samalyot va Kemani klassifikatsiya qiluvchi model')

# rasm yuklash
file = st.file_uploader('Rasm yuklang:', type=['png', 'jpeg'])

if file:
    st.image(file)
    
    #PIL conver
    img = PILImage.create(file)

    # modelni chaqirish
    model = load_learner('modelll.pkl')

    # bashorat qilish
    pred, pred_id, probs = model.predict(img)

    # natijani chiqarish
    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")

