import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io
from pathlib import Path


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_DIR = Path("Models")

MODELS = {
    "YOLO11n-Seg": MODEL_DIR / "best.pt",
}
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Vehicle Damage Inspection",
    page_icon="🚗",
    layout="wide"
)


# --------------------------------------------------
# Styling
# --------------------------------------------------

st.title("Vehicle Damage Inspection")

st.markdown(
    """
    Upload a vehicle-panel image and select a trained model to
    detect and segment **scratches** and **dents**.
    """
)


# --------------------------------------------------
# Model loading
# --------------------------------------------------

@st.cache_resource
def load_model(model_path):
    return YOLO(str(model_path))


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("Model Configuration")

selected_model = st.sidebar.selectbox(
    "Select Model",
    list(MODELS.keys())
)

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.25,
    step=0.05
)

model_path = MODELS[selected_model]


# --------------------------------------------------
# Validate model
# --------------------------------------------------

if not model_path.exists():
    st.error(
        f"Model file not found:\n\n`{model_path}`\n\n"
        "Please place the trained .pt file inside the models folder."
    )
    st.stop()


model = load_model(model_path)


# --------------------------------------------------
# Image upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload vehicle image",
    type=["jpg", "jpeg", "png", "webp"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Input Image")

    st.image(
        image,
        use_container_width=True
    )


    # --------------------------------------------------
    # Run inference
    # --------------------------------------------------

    if st.button(
        "🔍 Inspect Vehicle",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            f"Running {selected_model}..."
        ):

            image_np = np.array(image)

            results = model.predict(
                source=image_np,
                conf=confidence,
                verbose=False
            )

            result = results[0]

            # YOLO plot() returns BGR numpy array
            annotated_image = result.plot()

            # Convert BGR -> RGB
            annotated_image = annotated_image[:, :, ::-1]

            annotated_pil = Image.fromarray(
                annotated_image
            )


        # --------------------------------------------------
        # Results
        # --------------------------------------------------

        st.subheader("Inspection Result")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### Original")

            st.image(
                image,
                use_container_width=True
            )

        with col2:

            st.markdown("### Model Prediction")

            st.image(
                annotated_pil,
                use_container_width=True
            )


        # --------------------------------------------------
        # Detection summary
        # --------------------------------------------------

        st.subheader("Detection Summary")

        if result.boxes is None or len(result.boxes) == 0:

            st.success("No damage detected.")

        else:

            names = result.names

            detections = []

            for i in range(len(result.boxes)):

                cls_id = int(
                    result.boxes.cls[i].item()
                )

                conf = float(
                    result.boxes.conf[i].item()
                )

                detections.append({
                    "Damage": names[cls_id],
                    "Confidence": f"{conf:.2%}"
                })

            st.table(detections)


        # --------------------------------------------------
        # Download result
        # --------------------------------------------------

        buffer = io.BytesIO()

        annotated_pil.save(
            buffer,
            format="PNG"
        )

        st.download_button(
            label="⬇️ Download Annotated Image",
            data=buffer.getvalue(),
            file_name=f"{selected_model}_result.png",
            mime="image/png",
            use_container_width=True
        )