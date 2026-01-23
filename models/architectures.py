"""
Model Architectures for Brain Tumor Classification

This module contains implementations of VGG19, ResNet50, EfficientNet-B0,
and an Ensemble model for brain tumor classification.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BrainTumorClassifier(nn.Module):
    """
    Base classifier for brain tumor classification using pre-trained models.
    """
    
    def __init__(
        self,
        model_name: str,
        num_classes: int = 4,
        pretrained: bool = True,
        freeze_features: bool = False
    ):
        """
        Initialize the classifier.
        
        Args:
            model_name: Name of the model ('vgg19', 'resnet50', 'efficientnet_b0')
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            freeze_features: Whether to freeze feature extraction layers
        """
        super(BrainTumorClassifier, self).__init__()
        
        self.model_name = model_name.lower()
        self.num_classes = num_classes
        
        if self.model_name == 'vgg19':
            self.model = self._create_vgg19(pretrained, freeze_features)
        elif self.model_name == 'resnet50':
            self.model = self._create_resnet50(pretrained, freeze_features)
        elif self.model_name == 'efficientnet_b0':
            self.model = self._create_efficientnet_b0(pretrained, freeze_features)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        logger.info(f"Created {model_name} model with {num_classes} classes")
    
    def _create_vgg19(self, pretrained: bool, freeze_features: bool) -> nn.Module:
        """Create VGG19 model."""
        if pretrained:
            model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        else:
            model = models.vgg19(weights=None)
        
        # Freeze feature extraction layers if specified
        if freeze_features:
            for param in model.features.parameters():
                param.requires_grad = False
            logger.info("Froze VGG19 feature extraction layers")
        
        # Modify classifier
        num_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, self.num_classes)
        )
        
        return model
    
    def _create_resnet50(self, pretrained: bool, freeze_features: bool) -> nn.Module:
        """Create ResNet50 model."""
        if pretrained:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            model = models.resnet50(weights=None)
        
        # Freeze early layers if specified
        if freeze_features:
            # Freeze all layers except the last residual block
            for name, param in model.named_parameters():
                if 'layer4' not in name and 'fc' not in name:
                    param.requires_grad = False
            logger.info("Froze ResNet50 early layers")
        
        # Modify final fully connected layer
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes)
        )
        
        return model
    
    def _create_efficientnet_b0(self, pretrained: bool, freeze_features: bool) -> nn.Module:
        """Create EfficientNet-B0 model."""
        if pretrained:
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            model = models.efficientnet_b0(weights=None)
        
        # Freeze feature extraction if specified
        if freeze_features:
            for param in model.features.parameters():
                param.requires_grad = False
            logger.info("Froze EfficientNet-B0 feature layers")
        
        # Modify classifier
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes)
        )
        
        return model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)
    
    def unfreeze_all(self):
        """Unfreeze all parameters for fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info(f"Unfroze all parameters in {self.model_name}")
    
    def get_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


class EnsembleModel:
    """
    Ensemble model using majority voting across multiple models.
    """
    
    def __init__(
        self,
        models: Dict[str, nn.Module],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        class_names: List[str] = None
    ):
        """
        Initialize ensemble model.
        
        Args:
            models: Dictionary of model_name -> model
            device: Device to run inference on
            class_names: List of class names
        """
        self.models = models
        self.device = device
        self.class_names = class_names or ["glioma", "meningioma", "notumor", "pituitary"]
        
        # Move all models to device and set to eval mode
        for model in self.models.values():
            model.to(device)
            model.eval()
        
        logger.info(f"Created ensemble with {len(models)} models: {list(models.keys())}")
    
    def predict(
        self,
        image: torch.Tensor,
        return_all_predictions: bool = False
    ) -> Tuple[int, float, Dict[str, float]]:
        """
        Predict class using majority voting.
        
        Args:
            image: Input image tensor (batch_size, channels, height, width)
            return_all_predictions: Whether to return individual model predictions
            
        Returns:
            Tuple of (predicted_class, confidence, class_probabilities)
            If return_all_predictions=True, also returns individual predictions
        """
        image = image.to(self.device)
        
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for model_name, model in self.models.items():
                # Get prediction
                outputs = model(image)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                
                all_predictions.append(predicted_class)
                all_probabilities.append(probabilities.cpu().numpy()[0])
        
        # Majority voting
        prediction_counts = {}
        for pred in all_predictions:
            prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
        
        # Find the most common prediction
        max_count = max(prediction_counts.values())
        candidates = [pred for pred, count in prediction_counts.items() if count == max_count]
        
        if len(candidates) == 1:
            # Clear winner
            final_prediction = candidates[0]
        else:
            # Tie - use average probability to break
            avg_probabilities = torch.tensor(all_probabilities).mean(dim=0)
            final_prediction = torch.argmax(avg_probabilities).item()
        
        # Calculate confidence (average probability for predicted class)
        avg_probabilities = torch.tensor(all_probabilities).mean(dim=0)
        confidence = avg_probabilities[final_prediction].item()
        
        # Get all class probabilities
        class_probabilities = {
            self.class_names[i]: float(avg_probabilities[i])
            for i in range(len(self.class_names))
        }
        
        if return_all_predictions:
            individual_predictions = {
                model_name: {
                    'class': all_predictions[i],
                    'class_name': self.class_names[all_predictions[i]],
                    'probabilities': all_probabilities[i]
                }
                for i, model_name in enumerate(self.models.keys())
            }
            return final_prediction, confidence, class_probabilities, individual_predictions
        
        return final_prediction, confidence, class_probabilities
    
    def predict_batch(
        self,
        images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict classes for a batch of images.
        
        Args:
            images: Batch of images (batch_size, channels, height, width)
            
        Returns:
            Tuple of (predictions, confidences)
        """
        images = images.to(self.device)
        batch_size = images.shape[0]
        
        all_predictions = []
        
        with torch.no_grad():
            for model in self.models.values():
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                all_predictions.append(predictions)
        
        # Stack predictions (num_models, batch_size)
        all_predictions = torch.stack(all_predictions)
        
        # Majority voting for each sample
        final_predictions = []
        confidences = []
        
        for i in range(batch_size):
            sample_predictions = all_predictions[:, i].cpu().numpy()
            
            # Count votes
            prediction_counts = {}
            for pred in sample_predictions:
                prediction_counts[int(pred)] = prediction_counts.get(int(pred), 0) + 1
            
            # Get winner
            max_count = max(prediction_counts.values())
            candidates = [pred for pred, count in prediction_counts.items() if count == max_count]
            
            if len(candidates) == 1:
                final_prediction = candidates[0]
            else:
                # Tie - use first candidate (could be improved)
                final_prediction = candidates[0]
            
            final_predictions.append(final_prediction)
            confidences.append(max_count / len(self.models))
        
        return torch.tensor(final_predictions), torch.tensor(confidences)


def create_model(
    model_name: str,
    num_classes: int = 4,
    pretrained: bool = True,
    freeze_features: bool = False,
    device: str = None
) -> nn.Module:
    """
    Factory function to create a model.
    
    Args:
        model_name: Name of the model
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        freeze_features: Whether to freeze feature layers
        device: Device to move model to
        
    Returns:
        Created model
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = BrainTumorClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_features=freeze_features
    )
    
    model = model.to(device)
    
    logger.info(f"Model {model_name} created on {device}")
    logger.info(f"Trainable parameters: {model.get_trainable_params():,}")
    logger.info(f"Total parameters: {model.get_total_params():,}")
    
    return model


def load_model(
    model_path: str,
    model_name: str,
    num_classes: int = 4,
    device: str = None
) -> nn.Module:
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        model_name: Name of the model architecture
        num_classes: Number of output classes
        device: Device to load model to
        
    Returns:
        Loaded model
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model architecture
    model = BrainTumorClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False
    )
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {model_path}")
    
    return model


if __name__ == "__main__":
    # Example usage
    print("Brain Tumor Model Architectures")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Create models
    model_names = ['vgg19', 'resnet50', 'efficientnet_b0']
    
    for name in model_names:
        print(f"\nCreating {name.upper()} model:")
        model = create_model(name, num_classes=4, pretrained=True, device=device)
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        print(f"Trainable params: {model.get_trainable_params():,}")
    
    # Create ensemble
    print("\n" + "=" * 50)
    print("Creating Ensemble Model:")
    models_dict = {
        'vgg19': create_model('vgg19', device=device),
        'resnet50': create_model('resnet50', device=device),
        'efficientnet_b0': create_model('efficientnet_b0', device=device)
    }
    
    ensemble = EnsembleModel(models_dict, device=device)
    
    # Test ensemble prediction
    dummy_input = torch.randn(1, 3, 224, 224)
    pred, conf, probs = ensemble.predict(dummy_input)
    print(f"\nEnsemble prediction: Class {pred}")
    print(f"Confidence: {conf:.4f}")
    print(f"Class probabilities: {probs}")
