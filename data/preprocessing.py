"""
Data Preprocessing Module for Brain Tumor Classification

This module contains all preprocessing functions for loading, augmenting,
and preparing MRI images for the brain tumor classification models.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BrainTumorDataset(Dataset):
    """Custom Dataset for Brain Tumor MRI Images."""
    
    def __init__(
        self,
        root_dir: str,
        transform: transforms.Compose = None,
        class_names: List[str] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Root directory containing class folders
            transform: Torchvision transforms to apply
            class_names: List of class names
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.class_names = class_names or ["glioma", "meningioma", "notumor", "pituitary"]
        
        self.images = []
        self.labels = []
        
        # Load all images and labels
        for idx, class_name in enumerate(self.class_names):
            class_path = self.root_dir / class_name
            if not class_path.exists():
                logger.warning(f"Class directory {class_path} not found!")
                continue
                
            for img_path in class_path.glob("*.jpg"):
                self.images.append(str(img_path))
                self.labels.append(idx)
        
        logger.info(f"Loaded {len(self.images)} images from {root_dir}")
        
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def crop_brain_region(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Crop the brain region from MRI scan by removing black background.
    
    Args:
        image: Input image as numpy array
        threshold: Threshold for detecting brain region
        
    Returns:
        Cropped image containing only brain region
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Threshold to create binary mask
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image
    
    # Get bounding box of largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Add small padding
    padding = 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2 * padding)
    h = min(image.shape[0] - y, h + 2 * padding)
    
    # Crop image
    cropped = image[y:y+h, x:x+w]
    
    return cropped


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get training data augmentation transforms.
    
    Args:
        image_size: Target image size
        
    Returns:
        Composed transforms for training
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_val_test_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get validation/test transforms (no augmentation).
    
    Args:
        image_size: Target image size
        
    Returns:
        Composed transforms for validation/testing
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def preprocess_image(image_path: str, image_size: int = 224) -> torch.Tensor:
    """
    Preprocess a single image for inference.
    
    Args:
        image_path: Path to the image file
        image_size: Target size for the image
        
    Returns:
        Preprocessed image tensor ready for model input
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Apply transforms
    transform = get_val_test_transforms(image_size)
    image_tensor = transform(image)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor


def preprocess_pil_image(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    """
    Preprocess a PIL Image for inference.
    
    Args:
        image: PIL Image object
        image_size: Target size for the image
        
    Returns:
        Preprocessed image tensor ready for model input
    """
    # Ensure RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    transform = get_val_test_transforms(image_size)
    image_tensor = transform(image)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor


def load_and_split_dataset(
    data_path: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    image_size: int = 224,
    seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load dataset and split into train, validation, and test sets.
    
    Args:
        data_path: Path to dataset root directory
        test_size: Proportion of data for testing
        val_size: Proportion of training data for validation
        image_size: Target image size
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load full dataset with training transforms
    train_transform = get_train_transforms(image_size)
    val_test_transform = get_val_test_transforms(image_size)
    
    # Check if we have Training and Testing directories
    training_path = Path(data_path) / "Training"
    testing_path = Path(data_path) / "Testing"
    
    if training_path.exists() and testing_path.exists():
        logger.info("Found separate Training and Testing directories")
        
        # Load training data
        full_train_dataset = BrainTumorDataset(
            root_dir=str(training_path),
            transform=train_transform
        )
        
        # Split training into train and validation
        train_size = int((1 - val_size) * len(full_train_dataset))
        val_size_actual = len(full_train_dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            full_train_dataset,
            [train_size, val_size_actual],
            generator=torch.Generator().manual_seed(seed)
        )
        
        # Load test data
        test_dataset = BrainTumorDataset(
            root_dir=str(testing_path),
            transform=val_test_transform
        )
        
        # Update validation dataset transform
        val_dataset.dataset.transform = val_test_transform
        
    else:
        logger.info("Single directory found, splitting into train/val/test")
        
        # Load full dataset
        full_dataset = BrainTumorDataset(
            root_dir=data_path,
            transform=train_transform
        )
        
        # Calculate sizes
        total_size = len(full_dataset)
        test_size_actual = int(test_size * total_size)
        train_val_size = total_size - test_size_actual
        val_size_actual = int(val_size * train_val_size)
        train_size_actual = train_val_size - val_size_actual
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_size_actual, val_size_actual, test_size_actual],
            generator=torch.Generator().manual_seed(seed)
        )
    
    logger.info(f"Dataset split - Train: {len(train_dataset)}, "
                f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset


def create_data_loaders(
    train_data: Dataset,
    val_data: Dataset,
    test_data: Dataset,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for training, validation, and testing.
    
    Args:
        train_data: Training dataset
        val_data: Validation dataset
        test_data: Test dataset
        batch_size: Batch size for loading
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created DataLoaders with batch_size={batch_size}")
    
    return train_loader, val_loader, test_loader


def visualize_class_distribution(
    dataset: Dataset,
    class_names: List[str] = None,
    save_path: str = None
) -> None:
    """
    Visualize the class distribution in the dataset.
    
    Args:
        dataset: Dataset to visualize
        class_names: List of class names
        save_path: Optional path to save the plot
    """
    if class_names is None:
        class_names = ["glioma", "meningioma", "notumor", "pituitary"]
    
    # Get labels from dataset
    if hasattr(dataset, 'labels'):
        labels = dataset.labels
    elif hasattr(dataset, 'dataset'):
        # Handle Subset from random_split
        labels = [dataset.dataset.labels[i] for i in dataset.indices]
    else:
        logger.warning("Cannot extract labels from dataset")
        return
    
    # Count classes
    unique, counts = np.unique(labels, return_counts=True)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=[class_names[i] for i in unique], y=counts)
    plt.title('Class Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.xticks(rotation=45)
    
    # Add count labels on bars
    for i, count in enumerate(counts):
        plt.text(i, count, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved class distribution plot to {save_path}")
    
    plt.show()


def visualize_sample_images(
    dataset: Dataset,
    class_names: List[str] = None,
    samples_per_class: int = 3,
    save_path: str = None
) -> None:
    """
    Visualize sample images from each class.
    
    Args:
        dataset: Dataset to visualize
        class_names: List of class names
        samples_per_class: Number of samples to show per class
        save_path: Optional path to save the plot
    """
    if class_names is None:
        class_names = ["glioma", "meningioma", "notumor", "pituitary"]
    
    num_classes = len(class_names)
    
    # Create subplot grid
    fig, axes = plt.subplots(
        num_classes,
        samples_per_class,
        figsize=(samples_per_class * 3, num_classes * 3)
    )
    
    # Find samples for each class
    for class_idx in range(num_classes):
        samples_found = 0
        dataset_idx = 0
        
        while samples_found < samples_per_class and dataset_idx < len(dataset):
            img, label = dataset[dataset_idx]
            
            if label == class_idx:
                # Denormalize image for display
                img_display = img.clone()
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_display = img_display * std + mean
                img_display = torch.clamp(img_display, 0, 1)
                
                # Convert to numpy and transpose
                img_np = img_display.permute(1, 2, 0).numpy()
                
                # Plot
                ax = axes[class_idx, samples_found] if num_classes > 1 else axes[samples_found]
                ax.imshow(img_np)
                ax.axis('off')
                
                if samples_found == 0:
                    ax.set_title(f'{class_names[class_idx]}', fontsize=12, fontweight='bold')
                
                samples_found += 1
            
            dataset_idx += 1
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved sample images plot to {save_path}")
    
    plt.show()


def denormalize_image(
    tensor: torch.Tensor,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> np.ndarray:
    """
    Denormalize a tensor image for visualization.
    
    Args:
        tensor: Normalized image tensor
        mean: Mean used for normalization
        std: Std used for normalization
        
    Returns:
        Denormalized image as numpy array
    """
    tensor = tensor.clone()
    
    # Denormalize
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    
    # Clamp to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy
    img_np = tensor.permute(1, 2, 0).numpy()
    
    return img_np


if __name__ == "__main__":
    # Example usage
    print("Brain Tumor Dataset Preprocessing Module")
    print("=" * 50)
    
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Load and split dataset
    train_dataset, val_dataset, test_dataset = load_and_split_dataset(
        data_path=config['dataset']['data_path'],
        test_size=config['dataset']['test_size'],
        val_size=config['dataset']['val_size'],
        image_size=config['dataset']['image_size'],
        seed=config['seed']
    )
    
    # Visualize class distribution
    print("\nVisualizing class distribution...")
    visualize_class_distribution(train_dataset, config['dataset']['class_names'])
    
    # Visualize sample images
    print("\nVisualizing sample images...")
    visualize_sample_images(train_dataset, config['dataset']['class_names'])
