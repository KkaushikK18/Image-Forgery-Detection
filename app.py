import streamlit as st
import os
import sys
from PIL import Image
import tempfile
import torch

# Add src to sys path so we can import modules
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, "src"))

from predict import load_model, predict_image
from utils import get_device, compute_ela

# Configure page
st.set_page_config(page_title="Image Forgery Detection", page_icon="🔍", layout="wide")

# Constants
MODEL_PATH = os.path.join(project_root, "outputs", "models", "best_model.pth")

@st.cache_resource
def load_forgery_model():
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    device = get_device()
    model, backbone, in_channels = load_model(MODEL_PATH, device=device)
    return model, device, in_channels

def main():
    st.title("🔍 Image Forgery Detection AI")
    st.write("Upload an image to determine whether it is Authentic or Forged using our deep learning model based on Error Level Analysis (ELA).")

    model, device, in_channels = load_forgery_model()

    if model is None:
        st.error(f"Cannot find trained model at `{MODEL_PATH}`. Please ensure the path exists.")
        st.stop()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read image
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, use_container_width=True)
            
        with col2:
            st.subheader("Error Level Analysis (ELA)")
            ela_img = compute_ela(image)
            st.image(ela_img, use_container_width=True)
            st.caption("ELA highlights regions that have undergone different compression levels. Forged regions often stand out.")
            
        if st.button("Run Prediction", type="primary", use_container_width=True):
            with st.spinner("Analyzing image..."):
                # Save to temporary file since predict_image expects a path
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tf:
                    image.save(tf.name)
                    temp_path = tf.name
                
                try:
                    pred_class, confidence, class_name, input_tensor = predict_image(
                        model=model, 
                        image_path=temp_path, 
                        device=device, 
                        in_channels=in_channels
                    )
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("Prediction Results")
                    
                    if class_name == "Authentic":
                        st.success(f"**Result: AUTHENTIC**")
                        st.info(f"Confidence score: {confidence:.2f}%")
                    else:
                        st.error(f"**Result: FORGED**")
                        st.warning(f"Confidence score: {confidence:.2f}%")
                        
                    st.progress(confidence / 100.0)
                    
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

if __name__ == "__main__":
    main()
