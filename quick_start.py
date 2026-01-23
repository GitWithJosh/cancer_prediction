#!/usr/bin/env python3
"""
Quick Start Script for Brain Tumor Classification System

This script demonstrates the basic usage of the system.
Run this after installing dependencies to verify everything works.

Usage:
    python quick_start.py
"""

import sys
from pathlib import Path
import torch
import yaml

print("=" * 60)
print("Brain Tumor Classification System - Quick Start")
print("=" * 60)

# Check Python version
print(f"\n1. Python Version: {sys.version}")
if sys.version_info < (3, 8):
    print("   ❌ Error: Python 3.8+ required")
    sys.exit(1)
else:
    print("   ✓ Python version OK")

# Check PyTorch
print("\n2. Checking PyTorch...")
try:
    print(f"   PyTorch Version: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print("   ✓ PyTorch OK")
except Exception as e:
    print(f"   ❌ Error with PyTorch: {e}")
    sys.exit(1)

# Check config file
print("\n3. Checking configuration...")
config_path = Path("config.yaml")
if not config_path.exists():
    print("   ❌ config.yaml not found!")
    sys.exit(1)
else:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print("   ✓ Configuration loaded")

# Check dataset
print("\n4. Checking dataset...")
data_path = Path(config['dataset']['data_path'])
if not data_path.exists():
    print(f"   ❌ Dataset directory '{data_path}' not found!")
    print(f"   Please ensure your dataset is in '{data_path}'")
else:
    # Check for Training/Testing directories
    training_dir = data_path / "Training"
    testing_dir = data_path / "Testing"
    
    if training_dir.exists() and testing_dir.exists():
        print("   ✓ Training and Testing directories found")
        
        # Count images per class
        for class_name in config['dataset']['class_names']:
            train_class_dir = training_dir / class_name
            test_class_dir = testing_dir / class_name
            
            if train_class_dir.exists():
                train_count = len(list(train_class_dir.glob("*.jpg")))
                test_count = len(list(test_class_dir.glob("*.jpg"))) if test_class_dir.exists() else 0
                print(f"     - {class_name}: {train_count} training, {test_count} testing images")
    else:
        print("   ⚠ Warning: Expected Training/Testing structure not found")
        print(f"   Looking for single-directory structure...")
        total_images = sum(len(list((data_path / cls).glob("*.jpg"))) 
                          for cls in config['dataset']['class_names'] 
                          if (data_path / cls).exists())
        if total_images > 0:
            print(f"   ✓ Found {total_images} images")
        else:
            print("   ❌ No images found!")

# Check modules
print("\n5. Checking custom modules...")
modules_to_check = [
    ("data.preprocessing", "load_and_split_dataset"),
    ("models.architectures", "BrainTumorClassifier"),
    ("training.trainer", "train_model"),
    ("evaluation.metrics", "evaluate_model")
]

all_modules_ok = True
for module_name, function_name in modules_to_check:
    try:
        module = __import__(module_name, fromlist=[function_name])
        getattr(module, function_name)
        print(f"   ✓ {module_name}.{function_name}")
    except Exception as e:
        print(f"   ❌ {module_name}.{function_name}: {e}")
        all_modules_ok = False

if not all_modules_ok:
    print("\n❌ Some modules failed to import!")
    sys.exit(1)

# Test model creation
print("\n6. Testing model creation...")
try:
    from models.architectures import create_model
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_model = create_model('efficientnet_b0', num_classes=4, pretrained=False, device=device)
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        output = test_model(dummy_input)
    
    if output.shape == (1, 4):
        print(f"   ✓ Model creation successful (output shape: {output.shape})")
    else:
        print(f"   ❌ Unexpected output shape: {output.shape}")
except Exception as e:
    print(f"   ❌ Model creation failed: {e}")
    sys.exit(1)

# Check for trained models
print("\n7. Checking for trained models...")
models_dir = Path(config['paths'].get('models_dir', 'models'))
checkpoints_dir = Path(config['paths'].get('checkpoints_dir', 'checkpoints'))

model_files = list(models_dir.glob("*.pth")) if models_dir.exists() else []
checkpoint_files = list(checkpoints_dir.glob("*.pth")) if checkpoints_dir.exists() else []

if model_files or checkpoint_files:
    print(f"   ✓ Found {len(model_files)} models and {len(checkpoint_files)} checkpoints")
    for f in model_files[:3]:  # Show first 3
        print(f"     - {f.name}")
else:
    print("   ⚠ No trained models found")
    print("   Run 'python training/train_all_models.py' to train models")

# Summary
print("\n" + "=" * 60)
print("SYSTEM STATUS SUMMARY")
print("=" * 60)

status_items = [
    ("Python", "✓"),
    ("PyTorch", "✓"),
    ("Configuration", "✓"),
    ("Dataset", "✓" if data_path.exists() else "❌"),
    ("Modules", "✓" if all_modules_ok else "❌"),
    ("Model Creation", "✓"),
]

for item, status in status_items:
    print(f"{item:.<30} {status}")

print("\n" + "=" * 60)
print("NEXT STEPS")
print("=" * 60)

if not (model_files or checkpoint_files):
    print("""
1. Train models:
   python training/train_all_models.py

2. Or use individual notebooks:
   jupyter notebook 02_model_training.ipynb

3. Start Streamlit app (with pretrained models):
   streamlit run app.py
""")
else:
    print("""
✓ System is ready!

To start the Streamlit web app:
   streamlit run app.py

To explore data:
   jupyter notebook 01_data_exploration.ipynb

To compare models:
   jupyter notebook 03_model_comparison.ipynb
""")

print("=" * 60)
