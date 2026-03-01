import streamlit as st
from PIL import Image

from config import MODEL_PATH, IMAGE_SIZE, CLASS_NAMES_PATH
from utils import load_model, preprocess_image, predict, load_class_names

@st.cache_resource
def load_cached_model():
    return load_model(MODEL_PATH)

st.set_page_config(
    page_title="Plant Disease Classification",
    layout="centered"
)

def load_cached_assets():
    model = load_model(MODEL_PATH)
    class_names = load_class_names(CLASS_NAMES_PATH)
    return model, class_names

model, class_names = load_cached_assets()

st.title("Plant Disease Classification")
st.write(
    "Upload a leaf image to classify whether the plant is healthy or diseased."
)

model = load_cached_model()

class_names = sorted([
    p.name for p in
    (MODEL_PATH.parents[1] / "dataset"
     / "New Plant Diseases Dataset(Augmented)"
     / "train").iterdir()
    if p.is_dir()
])

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=400)

    img_array = preprocess_image(image, IMAGE_SIZE)

    with st.spinner("Analyzing leaf image..."):
        label, confidence = predict(model, img_array, class_names)

    st.subheader("Prediction Result")
    st.success(f"**{label}**")
    st.progress(int(confidence * 100))
    st.write(f"Confidence: **{confidence:.2%}**")
