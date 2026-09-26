import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

# -------------------------------------------------
# Page setup
# -------------------------------------------------
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="✏️",
    layout="centered"
)

# Custom styling
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .subtitle {
        text-align: center;
        color: #666;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .upload-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-bottom: 20px;
    }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        background-color: #f5f7fa;
        text-align: center;
        margin-top: 20px;
    }

    .digit {
        font-size: 64px;
        font-weight: 700;
        margin: 5px 0;
    }

    .confidence {
        font-size: 18px;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown(
    '<div class="main-title">MNIST Digit Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload a handwritten digit image and let the neural network predict it.'
    '</div>',
    unsafe_allow_html=True
)


# -------------------------------------------------
# Load the trained model
# -------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mnist_model.h5")
    return model


model = load_model()


# -------------------------------------------------
# Helper: preprocess image
# -------------------------------------------------
def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28))

    arr = np.array(img).astype("float32") / 255.0
    arr = arr.reshape(1, 784)

    return arr


# -------------------------------------------------
# Upload image
# -------------------------------------------------
st.markdown("### Upload your image")

uploaded_file = st.file_uploader(
    "Choose a PNG, JPG or JPEG image",
    type=["png", "jpg", "jpeg"]
)

image_to_predict = None

if uploaded_file is not None:

    image_to_predict = Image.open(uploaded_file)

    st.image(
        image_to_predict,
        caption="Uploaded Image",
        width=200
    )


# -------------------------------------------------
# Predict
# -------------------------------------------------
if st.button("Predict Digit", use_container_width=True):

    if image_to_predict is None:

        st.warning("Please upload a digit image first.")

    else:

        processed = preprocess_image(image_to_predict)

        prediction = model.predict(processed)

        predicted_digit = int(np.argmax(prediction))

        confidence = float(np.max(prediction)) * 100


        # -------------------------------------------------
        # Result
        # -------------------------------------------------
        st.markdown(
            '<div class="result-box">',
            unsafe_allow_html=True
        )

        st.markdown("### Prediction")

        st.markdown(
            f'<div class="digit">{predicted_digit}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="confidence">'
            f'Confidence: <strong>{confidence:.2f}%</strong>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown("</div>", unsafe_allow_html=True)


        # -------------------------------------------------
        # Prediction probabilities
        # -------------------------------------------------
        st.markdown("### Prediction Probabilities")

        st.bar_chart(prediction[0])