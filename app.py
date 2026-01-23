"""
Brain Tumor Classification - Streamlit Web Application

A complete web interface for brain tumor classification using deep learning.
Supports multiple models (VGG19, ResNet50, EfficientNet-B0) and ensemble prediction.
"""

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import time
from pathlib import Path
import sys
import random

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from models.architectures import BrainTumorClassifier, EnsembleModel
from data.preprocessing import preprocess_pil_image, denormalize_image
from evaluation.grad_cam import GradCAM

# Page configuration
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .tumor-detected {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .no-tumor {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 5px;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Class names and information
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

CLASS_INFO = {
    "glioma": {
        "emoji": "🔴",
        "name": "Glioma",
        "description": "**Glioma detected.** Gliomas are the most common type of malignant brain tumor. "
                      "They develop from glial cells and can be aggressive. Early detection and "
                      "treatment are crucial. Please consult with a healthcare professional immediately.",
        "severity": "high"
    },
    "meningioma": {
        "emoji": "🟠",
        "name": "Meningioma",
        "description": "**Meningioma detected.** Meningiomas are usually benign tumors that develop "
                      "from the membranes (meninges) surrounding the brain and spinal cord. While "
                      "typically slow-growing, they should be monitored by a medical professional.",
        "severity": "medium"
    },
    "pituitary": {
        "emoji": "🟡",
        "name": "Pituitary Tumor",
        "description": "**Pituitary tumor detected.** Pituitary tumors affect the pituitary gland and "
                      "can impact hormone production. Most are benign but may require treatment to "
                      "manage hormone levels. Medical consultation is recommended.",
        "severity": "medium"
    },
    "notumor": {
        "emoji": "🟢",
        "name": "No Tumor",
        "description": "**No tumor detected.** The brain scan appears healthy with no signs of tumor. "
                      "However, this AI system is a screening tool and not a replacement for "
                      "professional medical diagnosis.",
        "severity": "none"
    }
}

@st.cache_data
def get_test_images():
    """Get list of test images with their ground truth labels."""
    test_dir = Path("cancer_prediction_images/Testing")
    
    if not test_dir.exists():
        return {}
    
    test_images = {}
    
    for class_folder in ["glioma", "meningioma", "notumor", "pituitary"]:
        class_path = test_dir / class_folder
        if class_path.exists():
            images = list(class_path.glob("*.jpg")) + list(class_path.glob("*.png"))
            for img_path in images:
                test_images[str(img_path)] = class_folder
    
    return test_images


@st.cache_resource
def load_models():
    """Load all trained models. Uses Streamlit caching for efficiency."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models_dir = Path("checkpoints_10epochs_freeze_only_vgg")
    
    models = {}
    
    # Check if models exist
    model_files = {
        'vgg19': models_dir / 'vgg19_brain_tumor_best_state.pth',
        'resnet50': models_dir / 'resnet50_brain_tumor_best_state.pth',
        'efficientnet_b0': models_dir / 'efficientnet_b0_brain_tumor_best_state.pth'
    }
    
    available_models = {}
    
    for model_name, model_path in model_files.items():
        if model_path.exists():
            try:
                model = BrainTumorClassifier(
                    model_name=model_name,
                    num_classes=4,
                    pretrained=False
                )
                model.load_state_dict(torch.load(model_path, map_location=device))
                model = model.to(device)
                model.eval()
                available_models[model_name] = model
            except Exception as e:
                st.warning(f"Could not load {model_name}: {e}")
        else:
            # Create untrained model for demo purposes
            st.info(f"Model file not found: {model_path}. Using untrained {model_name} for demo.")
            model = BrainTumorClassifier(
                model_name=model_name,
                num_classes=4,
                pretrained=True
            )
            model = model.to(device)
            model.eval()
            available_models[model_name] = model
    
    # Create ensemble if multiple models available
    if len(available_models) >= 2:
        ensemble = EnsembleModel(available_models, device=device, class_names=CLASS_NAMES)
        models['ensemble'] = ensemble
    
    models.update(available_models)
    
    return models, device


def preprocess_for_display(image):
    """Preprocess image for display purposes."""
    # Resize for display
    display_image = image.copy()
    display_image.thumbnail((400, 400))
    return display_image


def predict_image(image, model_name, models, device):
    """
    Perform prediction on an image.
    
    Args:
        image: PIL Image
        model_name: Name of the model to use
        models: Dictionary of loaded models
        device: Device to run on
        
    Returns:
        Tuple of (prediction, confidence, all_probabilities, processing_time)
    """
    start_time = time.time()
    
    # Preprocess image
    image_tensor = preprocess_pil_image(image, image_size=224)
    
    # Get model
    if model_name == "Ensemble (Majority Voting)":
        if 'ensemble' in models:
            pred_idx, confidence, all_probs = models['ensemble'].predict(image_tensor)
            prediction = CLASS_NAMES[pred_idx]
        else:
            st.error("Ensemble model not available. Please select an individual model.")
            return None, None, None, None
    else:
        model_key = model_name.lower().replace('-', '_')
        if model_key in models:
            model = models[model_key]
            image_tensor = image_tensor.to(device)
            
            with torch.no_grad():
                output = model(image_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence_tensor, pred_idx_tensor = probabilities.max(1)
                
                pred_idx = pred_idx_tensor.item()
                confidence = confidence_tensor.item()
                prediction = CLASS_NAMES[pred_idx]
                
                # Get all class probabilities
                all_probs = {
                    CLASS_NAMES[i]: float(probabilities[0][i])
                    for i in range(len(CLASS_NAMES))
                }
        else:
            st.error(f"Model {model_name} not available.")
            return None, None, None, None
    
    processing_time = time.time() - start_time
    
    return prediction, confidence, all_probs, processing_time

def get_target_layer(model_name):
    """Get appropriate target layer for Grad-CAM based on model.
    
    For ResNet: We use the entire layer4 block (after all residual connections)
    which is the gold standard for ResNet + GradCAM.
    
    VGG19:
    model.features.34: Conv2d (last conv)
    model.features.35: ReLU  ← We choose this (after activation, before MaxPool)
    model.features.36: MaxPool2d  ← Loses spatial info

    ResNet50:
    model.layer4: Sequential  ← We choose this (entire final block, after residuals)
    model.avgpool: AdaptiveAvgPool2d  ← BLOCKS spatial info

    EfficientNet-B0:
    model.features.8: Sequential  ← We choose this (final block)
    model.avgpool: AdaptiveAvgPool2d  ← BLOCKS spatial info
    """
    layer_map = {
        "vgg19": "model.features.36",  # Last ReLU (after last conv, before MaxPool)
        "resnet50": "model.layer4",  # Entire layer4 block (gold standard for ResNet)
        "efficientnet_b0": "model.features.8",  # Final block
    }
    return layer_map.get(model_name.lower(), "model.features")


def generate_gradcam_visualization(image, model, model_name, pred_idx, device):
    """
    Generate GradCAM visualization for a given model and prediction.
    
    Args:
        image: PIL Image
        model: PyTorch model
        model_name: Name of the model (e.g., 'vgg19', 'resnet50')
        pred_idx: Predicted class index
        device: Device to run on
        
    Returns:
        PIL Image with GradCAM overlay
    """
    # Get target layer for this model
    target_layer = get_target_layer(model_name)
    
    # Create GradCAM instance
    grad_cam = GradCAM(model, target_layer, device)
    
    # Preprocess image
    image_tensor = preprocess_pil_image(image, image_size=224)
    
    # Generate heatmap
    heatmap = grad_cam.generate(image_tensor, pred_idx)
    
    # Overlay on original image
    gradcam_image = grad_cam.overlay_heatmap(image, heatmap, alpha=0.5)
    
    return gradcam_image

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<div class="main-header">🧠 Brain Tumor Classification System</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload an MRI scan for automatic tumor classification using AI</div>', 
                unsafe_allow_html=True)
    
    # Load models
    try:
        models, device = load_models()
        
        if not models:
            st.error("No models available. Please train models first or check the models directory.")
            st.stop()
            
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()
    
    # Sidebar
    st.sidebar.title("⚙️ Model Selection")
    
    # Model selection
    available_model_names = []
    if 'vgg19' in models:
        available_model_names.append("VGG19")
    if 'resnet50' in models:
        available_model_names.append("ResNet50")
    if 'efficientnet_b0' in models:
        available_model_names.append("EfficientNet-B0")
    if 'ensemble' in models:
        available_model_names.append("Ensemble (Majority Voting)")
    
    model_choice = st.sidebar.selectbox(
        "Choose Model",
        available_model_names,
        index=len(available_model_names) - 1 if available_model_names else 0,
        help="Select the AI model to use for classification"
    )
    
    st.sidebar.markdown("---")
    
    # About section
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.markdown("""
    This system classifies brain MRI scans into four categories:
    
    - 🔴 **Glioma**: Malignant brain tumor
    - 🟠 **Meningioma**: Usually benign tumor
    - 🟡 **Pituitary**: Pituitary gland tumor
    - 🟢 **No Tumor**: Healthy brain scan
    
    **Note:** This is an AI screening tool and not a substitute for professional medical diagnosis.
    """)
    
    st.sidebar.markdown("---")
    
    # Model info
    st.sidebar.subheader("📈 Model Information")
    if model_choice == "Ensemble (Majority Voting)":
        st.sidebar.info("🔗 Combines predictions from VGG19, ResNet50, and EfficientNet-B0 using majority voting for improved accuracy.")
    elif model_choice == "VGG19":
        st.sidebar.info("📊 VGG19: Deep convolutional network with 19 layers, pre-trained on ImageNet.")
    elif model_choice == "ResNet50":
        st.sidebar.info("📊 ResNet50: 50-layer residual network with skip connections for better gradient flow.")
    elif model_choice == "EfficientNet-B0":
        st.sidebar.info("📊 EfficientNet-B0: Efficient architecture with compound scaling for optimal performance.")
    
    st.sidebar.markdown(f"**Device:** {device.upper()}")
    
    # Main content
    st.markdown("---")
    
    # Image source selection
    st.subheader("📂 Select Image Source")
    image_source = st.radio(
        "Choose how to provide the MRI image:",
        ["Upload your own image", "Select from test dataset"],
        horizontal=True
    )
    
    uploaded_file = None
    image = None
    ground_truth = None
    
    if image_source == "Upload your own image":
        # File uploader
        uploaded_file = st.file_uploader(
            "📤 Choose an MRI image...",
            type=["jpg", "jpeg", "png"],
            help="Upload a brain MRI scan in JPG or PNG format"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
    
    else:  # Select from test dataset
        test_images = get_test_images()
        
        if not test_images:
            st.error("Test dataset not found. Please check that 'cancer_prediction_images/Testing' directory exists.")
        else:
            st.info(f"📊 {len(test_images)} test images available")
            
            # Filter by class
            col_filter1, col_filter2 = st.columns([1, 2])
            
            with col_filter1:
                filter_class = st.selectbox(
                    "Filter by class:",
                    ["All"] + ["glioma", "meningioma", "notumor", "pituitary"]
                )
            
            with col_filter2:
                # Filter images
                if filter_class == "All":
                    filtered_images = list(test_images.keys())
                else:
                    filtered_images = [path for path, label in test_images.items() if label == filter_class]
                
                if filtered_images:
                    # Random selection button
                    if st.button("🎲 Select Random Image", key="random_btn"):
                        st.session_state.selected_test_image = random.choice(filtered_images)
                    
                    # Dropdown selection
                    if 'selected_test_image' not in st.session_state or st.session_state.selected_test_image not in filtered_images:
                        st.session_state.selected_test_image = filtered_images[0]
                    
                    selected_image_path = st.selectbox(
                        f"Select image ({len(filtered_images)} available):",
                        filtered_images,
                        index=filtered_images.index(st.session_state.selected_test_image),
                        format_func=lambda x: Path(x).name
                    )
                    
                    st.session_state.selected_test_image = selected_image_path
                    
                    # Load image and ground truth
                    image = Image.open(selected_image_path).convert('RGB')
                    ground_truth = test_images[selected_image_path]
                else:
                    st.warning(f"No images found for class: {filter_class}")
    
    if image is not None:
        
        # Display images
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("🔍 Preprocessed Image")
            preprocessed_display = preprocess_for_display(image)
            st.image(preprocessed_display, use_column_width=True)
            st.caption("Image resized to 224×224 and normalized")
        
        st.markdown("---")
        
        # Prediction button
        col_button1, col_button2, col_button3 = st.columns([1, 2, 1])
        with col_button2:
            predict_button = st.button(
                "🔍 Classify Image",
                type="primary"
            )
        
        if predict_button:
            with st.spinner("🔄 Analyzing MRI scan..."):
                prediction, confidence, all_probs, proc_time = predict_image(
                    image, model_choice, models, device
                )
            
            if prediction is not None:
                # Success message
                st.success(f"✅ Analysis complete in {proc_time:.2f}s")
                
                # Ground truth comparison if available
                if ground_truth is not None:
                    st.markdown("---")
                    ground_truth_name = CLASS_INFO[ground_truth]['name']
                    is_correct = prediction == ground_truth
                    
                    if is_correct:
                        st.success(f"✅ **Correct Prediction!** Ground Truth: {CLASS_INFO[ground_truth]['emoji']} {ground_truth_name}")
                    else:
                        st.error(f"❌ **Incorrect Prediction.** Ground Truth: {CLASS_INFO[ground_truth]['emoji']} {ground_truth_name}")
                
                # Results section
                st.markdown("---")
                
                # Get class info
                class_info = CLASS_INFO[prediction]
                
                # Prediction box with color coding
                box_class = "no-tumor" if prediction == "notumor" else "tumor-detected"
                
                st.markdown(f'<div class="prediction-box {box_class}">', unsafe_allow_html=True)
                
                # Main prediction
                st.markdown("### 📊 Classification Result")
                
                col_pred1, col_pred2 = st.columns([2, 1])
                
                with col_pred1:
                    st.markdown(f"## {class_info['emoji']} **{class_info['name'].upper()}**")
                
                with col_pred2:
                    st.metric("Confidence", f"{confidence:.1%}")
                
                # Description
                st.markdown(class_info['description'])
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Class probabilities
                st.markdown("### 📈 Class Probabilities")
                
                # Sort probabilities by value
                sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
                
                for class_name, prob in sorted_probs:
                    col_prob1, col_prob2 = st.columns([3, 1])
                    
                    with col_prob1:
                        # Color code the progress bar
                        if class_name == prediction:
                            st.progress(prob, text=f"**{CLASS_INFO[class_name]['emoji']} {CLASS_INFO[class_name]['name']}**")
                        else:
                            st.progress(prob, text=f"{CLASS_INFO[class_name]['emoji']} {CLASS_INFO[class_name]['name']}")
                    
                    with col_prob2:
                        st.markdown(f"**{prob:.1%}**")
                
                # GradCAM Visualization
                st.markdown("---")
                st.markdown("### 🔥 GradCAM Visualization")
                st.markdown("GradCAM highlights the regions of the image that influenced the model's decision.")
                
                pred_idx = CLASS_NAMES.index(prediction)
                
                if model_choice == "Ensemble (Majority Voting)":
                    # Show GradCAM for all 3 models side by side
                    st.markdown("#### All Models")
                    
                    cols_gradcam = st.columns(3)
                    model_names = ['vgg19', 'resnet50', 'efficientnet_b0']
                    display_names = ['VGG19', 'ResNet50', 'EfficientNet-B0']
                    
                    for i, (model_name, display_name) in enumerate(zip(model_names, display_names)):
                        if model_name in models:
                            with cols_gradcam[i]:
                                with st.spinner(f"Generating {display_name} GradCAM..."):
                                    gradcam_img = generate_gradcam_visualization(
                                        image, 
                                        models[model_name], 
                                        model_name, 
                                        pred_idx, 
                                        device
                                    )
                                    st.image(gradcam_img, caption=display_name, use_column_width=True)
                else:
                    # Show GradCAM for single model
                    model_key = model_choice.lower().replace('-', '_')
                    
                    if model_key in models:
                        with st.spinner(f"Generating GradCAM for {model_choice}..."):
                            gradcam_img = generate_gradcam_visualization(
                                image, 
                                models[model_key], 
                                model_key, 
                                pred_idx, 
                                device
                            )
                            
                            col_gc1, col_gc2, col_gc3 = st.columns([1, 2, 1])
                            with col_gc2:
                                st.image(gradcam_img, caption=f"{model_choice} GradCAM", use_column_width=True)
                
                # Warning box
                st.markdown("---")
                st.warning("""
                ⚠️ **Medical Disclaimer**: This AI system is designed as a screening and educational tool. 
                It should NOT be used as the sole basis for medical decisions. Always consult with 
                qualified healthcare professionals for proper diagnosis and treatment.
                """)
                
    else:
        # Instructions when no file uploaded
        st.info("""
        👆 **How to use:**
        1. Select a model from the sidebar
        2. Upload a brain MRI scan image
        3. Click "Classify Image" to get the prediction
        
        The system will analyze the image and provide:
        - Classification result (tumor type or no tumor)
        - Confidence score
        - Probability distribution across all classes
        """)
        
        # Example placeholder
        st.markdown("---")
        st.subheader("📋 Supported Classes")
        
        cols = st.columns(4)
        for i, (class_key, info) in enumerate(CLASS_INFO.items()):
            with cols[i]:
                st.markdown(f"### {info['emoji']}")
                st.markdown(f"**{info['name']}**")
                if class_key == "notumor":
                    st.markdown("Healthy scan")
                else:
                    st.markdown("Tumor detected")


if __name__ == "__main__":
    main()
