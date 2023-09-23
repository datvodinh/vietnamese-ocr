import streamlit as st
import time
import torch
import os
from PIL import Image
from src.utils.inference import Inference
from streamlit_drawable_canvas import st_canvas

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
infer = Inference("checkpoint/model.pt")
st.title('OCR')
st.text('Author: Võ Đình Đạt')


st.write('***')
st.header("Draw Text: ")
old_option = None
# Specify canvas parameters in application
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform", "polygon", "point")
)

stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 5)
if drawing_mode == 'point':
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
realtime_update = st.sidebar.checkbox("Update in realtime", True)

    

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image= None,
    update_streamlit=realtime_update,
    height=256,
    width = 512,
    drawing_mode=drawing_mode,
    point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
    key="canvas",
)

st.write('***')
st.header("Result: ")
# print big prediction
if canvas_result.image_data is not None:
    col1,col2 = st.columns(2)
    img = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")
    with col1:
        st.image(img)
    with col2:
        target = infer.predict(img=img)
        st.markdown(f"""
            ```
            {target}
            ```
        """)
else:
    st.markdown(f"""
        ```
        Draw something!
        ```
    """)



