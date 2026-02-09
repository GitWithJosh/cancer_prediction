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
import json
import hashlib

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

# Custom CSS - Modern Minimalistic Design
st.markdown("""
<style>
    /* Main Layout */
    .main-header {
        font-size: 2.5rem;
        font-weight: 300;
        letter-spacing: -0.5px;
        text-align: center;
        color: #111;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        font-weight: 300;
        text-align: center;
        color: #666;
        margin-bottom: 3rem;
    }
    
    /* Prediction Box - Minimalistic */
    .prediction-box {
        padding: 2rem;
        border-radius: 12px;
        background-color: #fafafa;
        border: 1px solid #e0e0e0;
        margin: 1.5rem 0;
    }
    .tumor-detected {
        background-color: #fff5f5;
        border: 1px solid #ff6b6b;
    }
    .no-tumor {
        background-color: #f0fdf4;
        border: 1px solid #22c55e;
    }
    
    /* Clean Metrics */
    .metric-card {
        padding: 1.5rem;
        border-radius: 8px;
        background-color: #fff;
        border: 1px solid #e5e5e5;
    }
    
    /* Streamlit Component Overrides */
    .stButton>button {
        border-radius: 8px;
        font-weight: 400;
        transition: all 0.2s;
    }
    
    .stProgress > div > div > div {
        border-radius: 4px;
    }
    
    /* Hide Streamlit Branding for Cleaner Look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
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
def get_cache_filepath(model_name):
    """Get the cache file path for a specific model."""
    cache_dir = Path("prediction_cache")
    cache_dir.mkdir(exist_ok=True)
    
    # Sanitize model name for filename
    safe_model_name = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    return cache_dir / f"{safe_model_name}_predictions.json"


def load_predictions_cache(model_name):
    """
    Load predictions from disk cache if available.
    
    Returns:
        Dictionary of predictions or empty dict if cache doesn't exist
    """
    cache_file = get_cache_filepath(model_name)
    
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Could not load cache: {e}")
            return {}
    return {}


def save_predictions_cache(model_name, predictions):
    """
    Save predictions to disk cache.
    
    Args:
        model_name: Name of the model
        predictions: Dictionary of predictions to save
    """
    cache_file = get_cache_filepath(model_name)
    
    try:
        with open(cache_file, 'w') as f:
            json.dump(predictions, f, indent=2)
    except Exception as e:
        st.warning(f"Could not save cache: {e}")


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


def precompute_all_predictions(model_name, models, device, test_images):
    """
    Precompute predictions for all test images with the current model.
    Uses persistent cache on disk.
    
    Args:
        model_name: Name of the model to use
        models: Dictionary of loaded models
        device: Device to run on
        test_images: Dictionary of test image paths and their ground truth labels
        
    Returns:
        Dictionary mapping (image_path, model_key) to prediction
    """
    model_key = model_name.lower().replace('-', '_').replace(' (majority voting)', '').replace('ensemble', 'ensemble')
    
    # Try to load from persistent cache first
    cached_predictions = load_predictions_cache(model_name)
    
    # Check if we need to compute any new predictions
    predictions = {}
    images_to_compute = []
    
    for img_path in test_images.keys():
        cache_key = f"{img_path}_{model_key}"
        if cache_key in cached_predictions:
            predictions[cache_key] = cached_predictions[cache_key]
        else:
            images_to_compute.append(img_path)
    
    # Compute predictions for images not in cache
    if images_to_compute:
        progress_bar = st.progress(0, text=f"Computing {len(images_to_compute)} new predictions...")
        total = len(images_to_compute)
        
        for idx, img_path in enumerate(images_to_compute):
            try:
                temp_image = Image.open(img_path).convert('RGB')
                prediction, _, _, _, _ = predict_image(temp_image, model_name, models, device)
                cache_key = f"{img_path}_{model_key}"
                predictions[cache_key] = prediction
                
                # Update progress
                progress_bar.progress((idx + 1) / total, text=f"Computing predictions... {idx + 1}/{total}")
            except Exception as e:
                # Skip images that fail
                continue
        
        progress_bar.empty()
        
        # Save updated cache to disk
        save_predictions_cache(model_name, predictions)
    else:
        st.success(f"✓ Loaded {len(predictions)} predictions from cache")
    
    return predictions


def predict_image(image, model_name, models, device, return_individual_predictions=False):
    """
    Perform prediction on an image.
    
    Args:
        image: PIL Image
        model_name: Name of the model to use
        models: Dictionary of loaded models
        device: Device to run on
        return_individual_predictions: For ensemble, return individual model predictions
        
    Returns:
        Tuple of (prediction, confidence, all_probabilities, processing_time, individual_predictions)
        individual_predictions is None for non-ensemble models
    """
    start_time = time.time()
    
    # Preprocess image
    image_tensor = preprocess_pil_image(image, image_size=224)
    
    individual_predictions = None
    
    # Get model
    if model_name == "Ensemble (Majority Voting)":
        if 'ensemble' in models:
            if return_individual_predictions:
                pred_idx, confidence, all_probs, individual_predictions = models['ensemble'].predict(
                    image_tensor, return_all_predictions=True
                )
            else:
                pred_idx, confidence, all_probs = models['ensemble'].predict(image_tensor)
            prediction = CLASS_NAMES[pred_idx]
        else:
            st.error("Ensemble model not available. Please select an individual model.")
            return None, None, None, None, None
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
            return None, None, None, None, None
    
    processing_time = time.time() - start_time
    
    return prediction, confidence, all_probs, processing_time, individual_predictions

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
    
    # Header - Minimalistic
    st.markdown('<div class="main-header">Brain Tumor Classification</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered MRI analysis</div>', 
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
    
    # Sidebar - Minimalistic
    st.sidebar.title("Settings")
    
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
        "Model",
        available_model_names,
        index=len(available_model_names) - 1 if available_model_names else 0
    )
    
    # Check if model changed and trigger precomputation
    if 'last_model' not in st.session_state:
        st.session_state.last_model = None
    
    if st.session_state.last_model != model_choice:
        st.session_state.last_model = model_choice
        st.session_state.precompute_needed = True
    
    # Compact model info
    model_descriptions = {
        "Ensemble (Majority Voting)": "Combines all models",
        "VGG19": "19-layer CNN",
        "ResNet50": "50-layer residual network",
        "EfficientNet-B0": "Efficient scaling architecture"
    }
    st.sidebar.caption(model_descriptions.get(model_choice, ""))
    
    st.sidebar.markdown("---")
    
    # Classes info - Compact
    st.sidebar.subheader("Classes")
    st.sidebar.markdown("""
    🔴 Glioma  
    🟠 Meningioma  
    🟡 Pituitary  
    🟢 No Tumor
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Device: {device.upper()}")
    
    # Main content
    st.markdown("---")
    
    # Image source selection - Minimalistic
    image_source = st.radio(
        "Image Source",
        ["Upload", "Test Dataset"],
        horizontal=True
    )
    
    uploaded_file = None
    image = None
    ground_truth = None
    
    if image_source == "Upload":
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose MRI image",
            type=["jpg", "jpeg", "png"]
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
    
    else:  # Select from test dataset
        test_images = get_test_images()
        
        if not test_images:
            st.error("Test dataset not found.")
        else:
            # Initialize predictions cache in session state
            if 'predictions_cache' not in st.session_state:
                st.session_state.predictions_cache = {}
            
            # Precompute predictions if needed
            if st.session_state.get('precompute_needed', True):
                with st.spinner(f"Precomputing predictions for {len(test_images)} images with {model_choice}..."):
                    new_predictions = precompute_all_predictions(model_choice, models, device, test_images)
                    st.session_state.predictions_cache.update(new_predictions)
                    st.session_state.precompute_needed = False
            
            st.caption(f"{len(test_images)} images available")
            
            # Filter options - Compact layout
            col_filter1, col_filter2, col_filter3 = st.columns([1, 1, 1])
            
            with col_filter1:
                filter_class = st.selectbox(
                    "Class",
                    ["All", "glioma", "meningioma", "notumor", "pituitary"]
                )
            
            with col_filter2:
                # Filter for prediction correctness and disagreement
                filter_incorrect = st.selectbox(
                    "Prediction",
                    ["All", "Correct", "Incorrect", "All Models Disagree"]
                )
            
            with col_filter3:
                if st.button("🎲 Random", use_container_width=True):
                    st.session_state.trigger_random = True
            
            # Filter images by class
            if filter_class == "All":
                filtered_images = list(test_images.keys())
            else:
                filtered_images = [path for path, label in test_images.items() if label == filter_class]
            
            # Filter by prediction correctness (using precomputed predictions)
            if filter_incorrect != "All":
                # Get current model key
                model_key = model_choice.lower().replace('-', '_').replace(' (majority voting)', '').replace('ensemble', 'ensemble')
                
                # Filter based on precomputed predictions
                filtered_by_correctness = []
                
                if filter_incorrect == "All Models Disagree":
                    # Load predictions from all three individual models
                    vgg19_cache = load_predictions_cache("VGG19")
                    resnet50_cache = load_predictions_cache("ResNet50")
                    efficientnet_cache = load_predictions_cache("EfficientNet-B0")
                    
                    for img_path in filtered_images:
                        vgg19_pred = vgg19_cache.get(f"{img_path}_vgg19")
                        resnet50_pred = resnet50_cache.get(f"{img_path}_resnet50")
                        efficientnet_pred = efficientnet_cache.get(f"{img_path}_efficientnet_b0")
                        
                        # Check if all three predictions exist and are all different
                        if vgg19_pred and resnet50_pred and efficientnet_pred:
                            if len({vgg19_pred, resnet50_pred, efficientnet_pred}) == 3:
                                filtered_by_correctness.append(img_path)
                else:
                    # Original correct/incorrect filtering
                    for img_path in filtered_images:
                        cache_key = f"{img_path}_{model_key}"
                        
                        # Get prediction from cache (should always be there after precomputation)
                        prediction = st.session_state.predictions_cache.get(cache_key)
                        if prediction is None:
                            continue  # Skip if somehow not precomputed
                        
                        true_label = test_images[img_path]
                        is_correct = prediction == true_label
                        
                        if filter_incorrect == "Correct" and is_correct:
                            filtered_by_correctness.append(img_path)
                        elif filter_incorrect == "Incorrect" and not is_correct:
                            filtered_by_correctness.append(img_path)
                
                filtered_images = filtered_by_correctness
            
            if filtered_images:
                # Random selection
                if 'trigger_random' in st.session_state and st.session_state.trigger_random:
                    st.session_state.selected_test_image = random.choice(filtered_images)
                    st.session_state.trigger_random = False
                
                # Dropdown selection
                if 'selected_test_image' not in st.session_state or st.session_state.selected_test_image not in filtered_images:
                    st.session_state.selected_test_image = filtered_images[0]
                
                selected_image_path = st.selectbox(
                    f"{len(filtered_images)} images",
                    filtered_images,
                    index=filtered_images.index(st.session_state.selected_test_image),
                    format_func=lambda x: Path(x).name
                )
                
                st.session_state.selected_test_image = selected_image_path
                
                # Load image and ground truth
                image = Image.open(selected_image_path).convert('RGB')
                ground_truth = test_images[selected_image_path]
            else:
                st.warning(f"No images match the selected filters.")
    
    if image is not None:
        
        # Display images - Minimalistic
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original**")
            st.image(image, use_column_width=True)
        
        with col2:
            st.markdown("**Preprocessed**")
            preprocessed_display = preprocess_for_display(image)
            st.image(preprocessed_display, use_column_width=True)
            st.caption("224×224, normalized")
        
        st.markdown("---")
        
        # Prediction button - Centered and minimalistic
        col_button1, col_button2, col_button3 = st.columns([1, 2, 1])
        with col_button2:
            predict_button = st.button(
                "Classify",
                type="primary",
                use_container_width=True
            )
        
        if predict_button:
            with st.spinner("Analyzing..."):
                prediction, confidence, all_probs, proc_time, individual_preds = predict_image(
                    image, model_choice, models, device, return_individual_predictions=True
                )
            
            if prediction is not None:
                # Cache the prediction
                if ground_truth is not None and image_source == "Test Dataset":
                    model_key = model_choice.lower().replace('-', '_').replace(' (majority voting)', '').replace('ensemble', 'ensemble')
                    cache_key = f"{st.session_state.get('selected_test_image', '')}_{model_key}"
                    st.session_state.predictions_cache[cache_key] = prediction
                
                # Success message - Minimalistic
                st.success(f"Completed in {proc_time:.2f}s")
                
                # Ground truth comparison if available - Compact
                if ground_truth is not None:
                    ground_truth_name = CLASS_INFO[ground_truth]['name']
                    is_correct = prediction == ground_truth
                    
                    if is_correct:
                        st.success(f"✓ Correct — Ground truth: {CLASS_INFO[ground_truth]['emoji']} {ground_truth_name}")
                    else:
                        st.error(f"✗ Incorrect — Ground truth: {CLASS_INFO[ground_truth]['emoji']} {ground_truth_name}")
                
                # Results section - Minimalistic
                st.markdown("---")
                
                # Get class info
                class_info = CLASS_INFO[prediction]
                
                # Prediction box with color coding
                box_class = "no-tumor" if prediction == "notumor" else "tumor-detected"
                
                st.markdown(f'<div class="prediction-box {box_class}">', unsafe_allow_html=True)
                
                # Main prediction - Clean layout
                col_pred1, col_pred2 = st.columns([3, 1])
                
                with col_pred1:
                    st.markdown(f"## {class_info['emoji']} {class_info['name']}")
                    st.caption(class_info['description'])
                
                with col_pred2:
                    st.metric("", f"{confidence:.0%}")
                    st.caption("Confidence")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show individual model predictions for Ensemble
                if model_choice == "Ensemble (Majority Voting)" and individual_preds is not None:
                    st.markdown("### Individual Model Predictions")
                    
                    cols_models = st.columns(3)
                    model_order = ['vgg19', 'resnet50', 'efficientnet_b0']
                    display_names = {'vgg19': 'VGG19', 'resnet50': 'ResNet50', 'efficientnet_b0': 'EfficientNet-B0'}
                    
                    for idx, model_key in enumerate(model_order):
                        if model_key in individual_preds:
                            pred_info = individual_preds[model_key]
                            pred_class_name = pred_info['class_name']
                            pred_class_info = CLASS_INFO[pred_class_name]
                            pred_confidence = pred_info['probabilities'][pred_info['class']]
                            
                            with cols_models[idx]:
                                # Check if this model agreed with ensemble
                                agreed = pred_class_name == prediction
                                border_color = "#22c55e" if agreed else "#ff6b6b"
                                
                                st.markdown(f"""
                                <div style="
                                    padding: 1rem;
                                    border-radius: 8px;
                                    border: 2px solid {border_color};
                                    background-color: #fafafa;
                                    margin-bottom: 0.5rem;
                                ">
                                    <div style="font-weight: 600; font-size: 0.9rem; color: #666;">{display_names[model_key]}</div>
                                    <div style="font-size: 1.5rem; margin: 0.5rem 0;">{pred_class_info['emoji']}</div>
                                    <div style="font-weight: 500;">{pred_class_info['name']}</div>
                                    <div style="font-size: 1.2rem; font-weight: 600; color: #111; margin-top: 0.5rem;">{pred_confidence:.0%}</div>
                                </div>
                                """, unsafe_allow_html=True)
                
                # Class probabilities - Minimalistic
                st.markdown("### Probabilities")
                
                # Sort probabilities by value
                sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
                
                for class_name, prob in sorted_probs:
                    col_prob1, col_prob2 = st.columns([4, 1])
                    
                    with col_prob1:
                        label = f"{CLASS_INFO[class_name]['emoji']} {CLASS_INFO[class_name]['name']}"
                        if class_name == prediction:
                            st.progress(prob, text=f"**{label}**")
                        else:
                            st.progress(prob, text=label)
                    
                    with col_prob2:
                        st.markdown(f"{prob:.0%}")
                
                # GradCAM Visualization - Minimalistic
                st.markdown("---")
                st.markdown("### Attention Map")
                st.caption("Regions that influenced the decision")
                
                pred_idx = CLASS_NAMES.index(prediction)
                
                if model_choice == "Ensemble (Majority Voting)":
                    # Show GradCAM for all 3 models side by side
                    cols_gradcam = st.columns(3)
                    model_names = ['vgg19', 'resnet50', 'efficientnet_b0']
                    display_names = ['VGG19', 'ResNet50', 'EfficientNet-B0']
                    
                    for i, (model_name, display_name) in enumerate(zip(model_names, display_names)):
                        if model_name in models:
                            with cols_gradcam[i]:
                                with st.spinner(f"{display_name}..."):
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
                        with st.spinner("Generating..."):
                            gradcam_img = generate_gradcam_visualization(
                                image, 
                                models[model_key], 
                                model_key, 
                                pred_idx, 
                                device
                            )
                            
                            col_gc1, col_gc2, col_gc3 = st.columns([1, 2, 1])
                            with col_gc2:
                                st.image(gradcam_img, use_column_width=True)
                
                # Warning box - Minimalistic
                st.markdown("---")
                st.caption("⚠️ This is a screening tool. Consult medical professionals for diagnosis.")
                
    else:
        # Instructions when no file uploaded - Minimalistic
        st.info("""
        **How to use:**  
        1. Select a model  
        2. Choose an image source  
        3. Click Classify
        """)
        
        # Example placeholder - Compact
        st.markdown("---")
        st.markdown("**Classes**")
        
        cols = st.columns(4)
        for i, (class_key, info) in enumerate(CLASS_INFO.items()):
            with cols[i]:
                st.markdown(f"### {info['emoji']}")
                st.caption(info['name'])


if __name__ == "__main__":
    main()
