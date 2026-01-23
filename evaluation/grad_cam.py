import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2


class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) for visualizing
    model predictions on brain MRI images.
    """

    def __init__(self, model, target_layer, device):
        """
        Args:
            model: PyTorch model
            target_layer: Name of the layer to compute gradients from
            device: Device to run on (cuda or cpu)
        """
        self.model = model
        self.device = device
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        target = self._get_layer_by_name(self.target_layer)

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target.register_forward_hook(forward_hook)
        target.register_backward_hook(backward_hook)

    def _get_layer_by_name(self, layer_name):
        """Get layer from model by name."""
        # Support nested layer names like 'model.features.34'
        layer_parts = layer_name.split('.')
        module = self.model
        
        for part in layer_parts:
            if part.isdigit():
                # Handle numeric indices (e.g., features.34)
                module = module[int(part)]
            else:
                # Handle named attributes
                if hasattr(module, part):
                    module = getattr(module, part)
                else:
                    raise ValueError(f"Layer {layer_name} not found in model (failed at '{part}')")
        
        return module

    def generate(self, image_tensor, class_idx):
        """
        Generate Grad-CAM heatmap.

        Args:
            image_tensor: Input image tensor (1, 3, H, W)
            class_idx: Target class index

        Returns:
            Grad-CAM heatmap (H, W)
        """
        # Forward pass
        self.model.eval()
        image_tensor = image_tensor.to(self.device)

        output = self.model(image_tensor)
        target_score = output[0, class_idx]

        # Backward pass
        self.model.zero_grad()
        target_score.backward()

        # Compute Grad-CAM
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        # Weight activations by gradients
        weights = gradients.mean(dim=[1, 2])  # (C,)
        weighted_activations = weights.view(-1, 1, 1) * activations  # (C, H, W)

        # Create heatmap
        heatmap = weighted_activations.sum(dim=0).cpu().numpy()  # (H, W)
        heatmap = np.maximum(heatmap, 0)  # ReLU

        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap

    def overlay_heatmap(self, original_image, heatmap, alpha=0.5):
        """
        Overlay heatmap on original image.

        Args:
            original_image: PIL Image
            heatmap: Grad-CAM heatmap (H, W)
            alpha: Transparency of overlay

        Returns:
            PIL Image with overlay
        """
        # Convert PIL to numpy
        img_np = np.array(original_image)

        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))

        # Convert heatmap to color
        heatmap_color = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # Blend images
        overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap_color, alpha, 0)

        return Image.fromarray(overlay.astype(np.uint8))