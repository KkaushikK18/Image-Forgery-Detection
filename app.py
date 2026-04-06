import streamlit as st
import os
import sys
from PIL import Image
import tempfile
import torch
import cv2
import numpy as np
import plotly.graph_objects as go
import glob

# Add src to sys path so we can import modules
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, "src"))

from predict import load_model, predict_image
from utils import get_device, compute_ela, GradCAM, overlay_gradcam, analyze_exif

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

    st.sidebar.title("🎛️ Demo Controls")
    input_method = st.sidebar.radio("Image Input Method:", ["Upload your own Image", "Select from Demo Gallery"])
    
    raw_image = None
    if input_method == "Upload your own Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            raw_image = Image.open(uploaded_file)
            uploaded_file.seek(0)
    else:
        st.sidebar.subheader("🖼️ Sample Gallery")
        demo_files = glob.glob(os.path.join(project_root, "demo_images", "*.*"))
        demo_filenames = sorted([os.path.basename(f) for f in demo_files])
        if not demo_filenames:
            st.warning("No images found in demo_images/")
        else:
            selected_file = st.sidebar.selectbox("Choose a sample image:", demo_filenames)
            if selected_file:
                raw_image = Image.open(os.path.join(project_root, "demo_images", selected_file))
                
    if raw_image is not None:
        try:
            exif_dict, exif_warnings = analyze_exif(raw_image)
            image = raw_image.convert("RGB")
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return
            
        # ── EXIF Forensics UI ──
        st.markdown("---")
        st.subheader("🕵️ Metadata Forensics (EXIF)")
        
        if exif_warnings:
            for w in exif_warnings:
                st.warning(f"**Anomaly Detected:** {w}", icon="⚠️")
        else:
            st.success("No obvious metadata anomalies detected (or no EXIF present).", icon="✅")
            
        if exif_dict:
            with st.expander("View Raw EXIF Metadata"):
                # Format to JSON for cleaner collapsible viewing
                clean_exif = {str(k): str(v) for k, v in exif_dict.items()}
                st.json(clean_exif)
                
        st.markdown("---")
        
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
                        st.success(f"🌿 **Result: AUTHENTIC** - No Forgery Detected")
                    else:
                        st.error(f"🚨 **Result: FORGED** - Manipulation Detected")
                        
                    # Create Plotly Gauge Chart
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = confidence,
                        number = {'suffix': "%"},
                        title = {'text': "Authenticity Confidence" if class_name == "Authentic" else "Forgery Confidence"},
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        gauge = {
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "#198754" if class_name == "Authentic" else "#dc3545"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 50], 'color': "rgba(25, 135, 84, 0.1)" if class_name == "Authentic" else "rgba(220, 53, 69, 0.1)"},
                                {'range': [50, 80], 'color': "rgba(25, 135, 84, 0.3)" if class_name == "Authentic" else "rgba(220, 53, 69, 0.3)"},
                                {'range': [80, 100], 'color': "rgba(25, 135, 84, 0.5)" if class_name == "Authentic" else "rgba(220, 53, 69, 0.5)"}
                            ]
                        }
                    ))
                    fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # ── Grad-CAM Integration ──
                    with st.spinner("Generating Explainability Heatmap (Grad-CAM)..."):
                        input_tensor.requires_grad_(True)
                        target_layer = model.get_gradcam_target_layer()
                        cam = GradCAM(model, target_layer)
                        heatmap = cam.generate(input_tensor, target_class=pred_class)
                        
                        overlay_bgr = overlay_gradcam(temp_path, heatmap)
                        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
                        
                        st.markdown("### 🧠 Model Interpretation")
                        st.caption("Grad-CAM highlights the pixel regions that contributed most to the model's decision.")
                        
                        gc1, gc2, gc3 = st.columns(3)
                        
                        with gc1:
                            orig = cv2.cvtColor(cv2.resize(cv2.imread(temp_path), (224, 224)), cv2.COLOR_BGR2RGB)
                            st.image(orig, caption="Model Input Size", use_container_width=True)
                        with gc2:
                            # Heatmap alone - map heatmap[0,1] to RGB
                            heatmap_color = cv2.applyColorMap(np.uint8(255 * cv2.resize(heatmap, (224, 224))), cv2.COLORMAP_JET)
                            st.image(cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB), caption="Raw Heatmap", use_container_width=True)
                        with gc3:
                            st.image(overlay_rgb, caption="Heatmap Overlay", use_container_width=True)
                    
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

if __name__ == "__main__":
    main()
