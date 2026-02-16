import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import joblib
import os

# --- Configuration & Setup ---
st.set_page_config(
    page_title="Chest X-ray Pneumonia Detection",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Custom CSS (Strict Single Screen) ---
st.markdown("""
<style>
    /* 1. Global Reset & Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #000000;
        background-color: #FFFFFF;
        overflow: hidden; /* Prevent body scrolling */
    }

    /* 2. Remove Streamlit Chrome */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* 3. Main Container Layout */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
        height: 100vh;
        overflow-y: hidden;
    }
    
    /* Force White Background on App Container */
    .stApp {
        background-color: #FFFFFF !important;
    }
    
    /* 4. Column Layout & Cards */
    /* We target the divs that hold the columns to make them fill height */
    [data-testid="column"] {
        height: 95vh;
        overflow-y: auto; /* Internal scroll if needed */
        padding: 0 10px;
        background-color: #FFFFFF;
    }

    .medical-card {
        background-color: #FFFFFF;
        border: 2px solid #000; /* Black Border */
        border-radius: 0px; /* Square/Professional look */
        padding: 20px;
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    
    /* 5. Typography */
    h1 {
        font-weight: 800;
        font-size: 2rem;
        color: #000;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: -0.5px;
    }
    h3 {
        font-weight: 600;
        font-size: 1rem;
        color: #555;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #000;
        padding-bottom: 10px;
    }
    
    /* 6. Components */
    /* Uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #000;
        padding: 30px;
        background-color: #FFFFFF; /* Strict White */
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        background-color: #000;
        color: #fff;
        border: none;
        border-radius: 4px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        padding: 0.8rem 1rem;
        margin-top: 20px;
    }
    .stButton > button:hover {
        background-color: #333;
        color: #fff;
    }
    
    /* Result Box */
    .result-box {
        margin-top: auto; /* Push to bottom */
        padding: 20px;
        border: 2px solid #000;
        text-align: center;
        background-color: #FFFFFF;
    }
    
    /* Image Styling */
    img {
        max-height: 60vh; /* Limit image height */
        object-fit: contain;
        border: 1px solid #ddd;
    }
    
    /* Force inputs/textareas to have white background */
    .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border-color: #000000 !important;
    }
    
</style>
""", unsafe_allow_html=True)

# --- Model Definition ---
class ENB4WithEmbeddings(nn.Module):
    def __init__(self, num_classes=2):
        super(ENB4WithEmbeddings, self).__init__()
        self.base_model = models.efficientnet_b4(weights=None)
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, num_classes, bias=True)
        )

    def forward(self, x):
        features = self.base_model.features(x)
        features = self.base_model.avgpool(features)
        embeddings = torch.flatten(features, 1)
        logits = self.base_model.classifier(embeddings)
        return logits, embeddings

# --- Load Models & Resources ---
@st.cache_resource
def load_all_models():
    models_dict = {}
    try:
        # 1. Load EfficientNetB4
        model_enb4 = ENB4WithEmbeddings(num_classes=2)
        state_dict = torch.load('efficientnetb4_model.pth', map_location=device, weights_only=False)
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('features.0.'):
                new_key = k.replace('features.0.', 'base_model.features.', 1)
                new_state_dict[new_key] = v
            elif k.startswith('fc.'):
                new_key = k.replace('fc.', 'base_model.classifier.', 1)
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        
        model_enb4.load_state_dict(new_state_dict, strict=False)
        model_enb4.to(device)
        model_enb4.eval()
        models_dict['enb4'] = model_enb4
        
        # 2. Other Models
        models_dict['xgboost'] = joblib.load('xgboost_model.joblib')
        models_dict['scaler'] = joblib.load('scaler.joblib')
        models_dict['pca'] = joblib.load('pca.joblib')
        models_dict['meta'] = joblib.load('meta_classifier.joblib')
        models_dict['transforms'] = torch.load('test_transforms.pth', map_location=device, weights_only=False)
        
        return models_dict
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# --- Prediction Pipeline ---
def predict_image(image, loaded_models):
    enb4 = loaded_models['enb4']
    xgb_model = loaded_models['xgboost']
    scaler = loaded_models['scaler']
    pca = loaded_models['pca']
    meta_clf = loaded_models['meta']
    transforms_pipeline = loaded_models['transforms']
    
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Manual ToTensor
    basic_transform = transforms.ToTensor()
    img_tensor = basic_transform(image)
    
    # Apply loaded transforms
    img_tensor = transforms_pipeline(img_tensor)
    
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device)
    
    # Inference
    with torch.no_grad():
        logits, embeddings = enb4(img_tensor)
        probs_net = torch.softmax(logits, dim=1)
        p_net = probs_net[0][1].item()
        embeddings_np = embeddings.cpu().numpy()
        
    embeddings_scaled = scaler.transform(embeddings_np)
    embeddings_pca = pca.transform(embeddings_scaled)
    p_xgb = xgb_model.predict_proba(embeddings_pca)[0][1]
    
    stacked_features = np.array([[p_net, p_xgb]])
    p_ens = meta_clf.predict_proba(stacked_features)[0][1]
    
    return p_ens, p_net, p_xgb

# --- Main Layout ---
def main():
    # Load models
    loaded_models = load_all_models()
    
    # 2-Column Split
    col1, col2 = st.columns([4, 3], gap="large")
    
    # Left: Input
    with col1:
        st.markdown("###  Patient Scan")
        uploaded_file = st.file_uploader("Select X-ray Image", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            # Display with standard size relative to column
            st.image(image, caption="Patient Radiograph", use_column_width=True)
        else:
            st.info("Awaiting Image Upload...")
            # Placeholder for consistent height
            st.markdown('<div style="height: 50vh; background: #fafafa; border: 1px solid #eee; display: flex; align-items: center; justify-content: center;">No Image Selected</div>', unsafe_allow_html=True)

    # Right: Analysis
    with col2:
        st.markdown("<h1>PNEUMONIA<br>DETECTION</h1>", unsafe_allow_html=True)
        st.markdown("<b>SYSTEM READY</b>", unsafe_allow_html=True)
        st.divider()
        
        if uploaded_file and loaded_models:
            if st.button("INITIATE ANALYSIS"):
                with st.spinner("Processing..."):
                    p_ens, p_net, p_xgb = predict_image(image, loaded_models)
                    
                    label = "PNEUMONIA" if p_ens >= 0.5 else "NORMAL"
                    conf_val = p_ens if label == "PNEUMONIA" else (1 - p_ens)
                    
                    # Result Display
                    st.markdown(f"""
                    <div class="result-box">
                        <div style="font-size: 1.5rem; color: #555;">DIAGNOSIS</div>
                        <div style="font-size: 3rem; font-weight: 800; color: {'#000' if label == 'NORMAL' else '#D80000'};">
                            {label}
                        </div>
                        <div style="margin-top: 10px; font-size: 1.2rem;">
                            CONFIDENCE: <b>{conf_val:.2%}</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Detailed Metrics
                    st.markdown("### Model Consensus")
                    st.progress(p_ens)
                    st.write(f"Ensemble Score: {p_ens:.4f}")
                    st.write(f"CNN Confidence: {p_net:.4f}")
                    st.write(f"XGB Confidence: {p_xgb:.4f}")

if __name__ == '__main__':
    main()
