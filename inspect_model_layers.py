"""
Script to inspect which layers are frozen/trainable in the models.
This helps verify that only the classification layers are being trained.

Usage:
    python inspect_model_layers.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import yaml
from models.architectures import create_model
from collections import OrderedDict


def count_parameters(model):
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    frozen = total - trainable
    return trainable, frozen, total


def inspect_model_layers(model, model_name):
    """Inspect and display which layers are frozen/trainable."""
    print(f"\n{'='*80}")
    print(f"Inspecting {model_name.upper()}")
    print(f"{'='*80}\n")
    
    # Get all named parameters
    trainable_layers = []
    frozen_layers = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_layers.append((name, param.numel()))
        else:
            frozen_layers.append((name, param.numel()))
    
    # Print frozen layers
    print(f"🔒 FROZEN LAYERS ({len(frozen_layers)} layers):")
    print("-" * 80)
    if frozen_layers:
        for name, num_params in frozen_layers[:10]:  # Show first 10
            print(f"  ❄️  {name:60s} {num_params:>12,} params")
        if len(frozen_layers) > 10:
            print(f"  ... and {len(frozen_layers) - 10} more frozen layers")
        frozen_total = sum(p for _, p in frozen_layers)
        print(f"\n  Total frozen parameters: {frozen_total:,}")
    else:
        print("  ⚠️  NO LAYERS ARE FROZEN!")
    
    # Print trainable layers
    print(f"\n🔓 TRAINABLE LAYERS ({len(trainable_layers)} layers):")
    print("-" * 80)
    if trainable_layers:
        for name, num_params in trainable_layers:
            print(f"  ✅ {name:60s} {num_params:>12,} params")
        trainable_total = sum(p for _, p in trainable_layers)
        print(f"\n  Total trainable parameters: {trainable_total:,}")
    else:
        print("  ⚠️  NO LAYERS ARE TRAINABLE!")
    
    # Summary
    trainable, frozen, total = count_parameters(model)
    print(f"\n📊 SUMMARY:")
    print("-" * 80)
    print(f"  Total parameters:      {total:>15,}")
    print(f"  Trainable parameters:  {trainable:>15,} ({100*trainable/total:>5.2f}%)")
    print(f"  Frozen parameters:     {frozen:>15,} ({100*frozen/total:>5.2f}%)")
    
    # Check if only classifier is trainable
    if model_name == 'vgg19':
        classifier_names = ['classifier', 'model.classifier']
    elif model_name == 'resnet50':
        classifier_names = ['fc', 'model.fc', 'layer4', 'model.layer4']
    elif model_name == 'efficientnet_b0':
        classifier_names = ['classifier', 'model.classifier']
    else:
        classifier_names = []
    
    non_classifier_trainable = []
    for name, _ in trainable_layers:
        if not any(clf_name in name for clf_name in classifier_names):
            non_classifier_trainable.append(name)
    
    print(f"\n⚠️  VERIFICATION:")
    print("-" * 80)
    if non_classifier_trainable:
        print(f"  ❌ WARNING: {len(non_classifier_trainable)} non-classifier layers are trainable!")
        print(f"     This may indicate incorrect freezing!")
        for name in non_classifier_trainable[:5]:
            print(f"     - {name}")
        if len(non_classifier_trainable) > 5:
            print(f"     ... and {len(non_classifier_trainable) - 5} more")
    else:
        print(f"  ✅ CORRECT: Only classifier/final layers are trainable!")
    
    return trainable, frozen, total


def main():
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Using device: {device}")
    
    model_names = ['vgg19', 'resnet50', 'efficientnet_b0']
    all_stats = {}
    
    for model_name in model_names:
        # Create model with same settings as training
        model = create_model(
            model_name=model_name,
            num_classes=config['dataset']['num_classes'],
            pretrained=config['models'][model_name]['pretrained'],
            freeze_features=config['models'][model_name].get('freeze_features', False),
            device=device
        )
        
        # Inspect layers
        trainable, frozen, total = inspect_model_layers(model, model_name)
        all_stats[model_name] = {
            'trainable': trainable,
            'frozen': frozen,
            'total': total
        }
    
    # Final comparison
    print(f"\n\n{'='*80}")
    print("FINAL COMPARISON")
    print(f"{'='*80}\n")
    print(f"{'Model':<20} {'Total Params':>15} {'Trainable':>15} {'Frozen':>15} {'% Trainable':>12}")
    print("-" * 80)
    
    for model_name, stats in all_stats.items():
        pct = 100 * stats['trainable'] / stats['total']
        print(f"{model_name:<20} {stats['total']:>15,} {stats['trainable']:>15,} {stats['frozen']:>15,} {pct:>11.2f}%")
    
    print("\n" + "="*80)
    print("EXPECTED BEHAVIOR (according to paper):")
    print("="*80)
    print("✅ VGG19:          Only classifier layers trainable (features frozen)")
    print("✅ ResNet50:       Only layer4 + fc trainable (early layers frozen)")
    print("✅ EfficientNet:   Only classifier trainable (features frozen)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
