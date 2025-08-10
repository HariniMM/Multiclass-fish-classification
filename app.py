import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

st.title("Fish Species Classification")

@st.cache_resource
def load_trained_model():
    model = load_model('best_resnet50_model.h5')
    return model

model = load_trained_model()

class_indices = {
    0: 'animal fish',
    1: 'animal fish bass',
    2: 'fish sea_food black_sea_sprat',
    3: 'fish sea_food gilt_head_bream',
    4: 'fish sea_food hourse_mackerel',
    5: 'fish sea_food red_mullet',
    6: 'fish sea_food red_sea_bream',
    7: 'fish sea_food sea_bass',
    8: 'fish sea_food shrimp',
    9: 'fish sea_food striped_red_mullet',
    10: 'fish sea_food trout'
}

uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption='Uploaded Image', use_container_width=True)
    
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # rescale
    
    preds = model.predict(x)
    predicted_class = np.argmax(preds[0])
    confidence = preds[0][predicted_class]
    
    st.write(f"**Predicted Fish Class:** {class_indices[predicted_class]}")
    st.write(f"**Confidence:** {confidence:.2f}")

