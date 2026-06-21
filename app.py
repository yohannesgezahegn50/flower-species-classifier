import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time


st.set_page_config(
    page_title="Flower Species Classifier",
    page_icon="🌸",
    layout="centered"
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "flower_classifier.keras"
    )

model = load_model()


class_names = [
    "dandelion",
    "daisy",
    "tulips",
    "sunflowers",
    "roses"
]


flower_info = {
    "daisy": "Daisies are known for their white petals and yellow centers.",
    "dandelion": "Dandelions are flowering plants commonly found in grasslands and gardens.",
    "roses": "Roses are popular ornamental flowers with many colors and varieties.",
    "sunflowers": "Sunflowers are famous for their large yellow petals and tendency to follow the sun.",
    "tulips": "Tulips are colorful spring-blooming flowers that originated in Central Asia."
}



st.title("🌸 Flower Species Classifier")

st.markdown(
    "Upload a flower image and let the AI identify its species."
)

st.divider()


col1, col2 = st.columns([3, 1])

with col1:
    uploaded_file = st.file_uploader(
        "📷 Upload Flower Image",
        type=["jpg", "jpeg", "png", "jfif", "webp"]
    )

with col2:
    st.write("")  # spacing
    st.write("")  # spacing

    predict_button = st.button(
        "🔍 Predict",
        use_container_width=True
    )


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width=350
    )

   
    if predict_button:

        with st.spinner(
            "🌸 Analyzing flower image..."
        ):

            img = image.resize((224, 224))
            img = np.array(img)
            img = np.expand_dims(img, axis=0)

            # Small delay for better UX
            time.sleep(1)

            prediction = model.predict(
                img,
                verbose=0
            )

        predicted_class = class_names[
            np.argmax(prediction)
        ]

        confidence = np.max(prediction)

        st.divider()

        st.subheader("Prediction Result")

        st.success(
            f"🌼 {predicted_class.capitalize()}"
        )

        st.metric(
            label="Confidence",
            value=f"{confidence:.2%}"
        )

        
        if confidence < 0.60:
            st.warning(
                "The model is not very confident about this prediction."
            )

        
        st.info(
            flower_info[predicted_class]
        )

        st.divider()

        st.subheader(
            "Class Probabilities"
        )

        for flower, prob in zip(
            class_names,
            prediction[0]
        ):

            st.write(
                f"**{flower.capitalize()}** : {prob:.2%}"
            )

            st.progress(
                float(prob)
            )

        st.balloons()


st.divider()

st.caption(
    "Built with TensorFlow, EfficientNetB0, and Streamlit 🚀"
)