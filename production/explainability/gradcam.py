#!/usr/bin/env python3
"""
Model Explainability using Grad-CAM
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))


class GradCAM:
    """Gradient-weighted Class Activation Mapping for model explainability"""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Initialize Grad-CAM
        
        Args:
            model: The neural network model
            target_layer: The target layer to compute CAM (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        """Hook to capture activations"""
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        """Hook to capture gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate CAM for input
        
        Args:
            input_tensor: Input tensor (1, C, H, W)
            target_class: Target class index (if None, uses predicted class)
            
        Returns:
            CAM heatmap as numpy array
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Generate CAM
        gradients = self.gradients[0]  # C x H x W
        activations = self.activations[0]  # C x H x W
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # C
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU
        cam = torch.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def visualize(self, image: np.ndarray, cam: np.ndarray, 
                 output_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize CAM overlay on image
        
        Args:
            image: Original image (BGR format)
            cam: CAM heatmap
            output_path: Optional path to save visualization
            
        Returns:
            Visualization as numpy array
        """
        # Resize CAM to match image size
        h, w = image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Convert to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Convert image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlay = cv2.addWeighted(image_rgb, 0.6, heatmap, 0.4, 0)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image_rgb)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"[INFO] Grad-CAM visualization saved to {output_path}")
        
        # Convert to numpy array
        fig.canvas.draw()
        vis = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis = vis.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return vis


class ExplainabilityEngine:
    """High-level interface for model explainability"""
    
    def __init__(self, model: nn.Module, device: str = "cpu"):
        """
        Initialize explainability engine
        
        Args:
            model: Model to explain
            device: Device to run on
        """
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        
        # Find target layer (last conv layer)
        self.target_layer = self._find_target_layer()
        
        # Initialize Grad-CAM
        if self.target_layer:
            self.gradcam = GradCAM(self.model, self.target_layer)
        else:
            self.gradcam = None
            print("[WARNING] Could not find suitable target layer for Grad-CAM")
    
    def _find_target_layer(self) -> Optional[nn.Module]:
        """Find the last convolutional layer in the model"""
        target_layer = None
        
        # Search for the last Conv2d layer
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                target_layer = module
        
        if target_layer:
            print(f"[INFO] Using target layer for Grad-CAM")
        
        return target_layer
    
    def explain_prediction(self, image_path: str, preprocess_fn,
                          target_class: Optional[int] = None,
                          output_path: Optional[str] = None) -> dict:
        """
        Generate explanation for a prediction
        
        Args:
            image_path: Path to input image
            preprocess_fn: Function to preprocess image
            target_class: Target class (if None, uses predicted class)
            output_path: Optional path to save visualization
            
        Returns:
            Dictionary with explanation results
        """
        if not self.gradcam:
            return {"error": "Grad-CAM not available"}
        
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        # Preprocess
        input_tensor = preprocess_fn(img).unsqueeze(0).to(self.device)
        
        # Get prediction
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
            pred_class = output.argmax(dim=1).item()
            confidence = torch.softmax(output, dim=1)[0, pred_class].item()
        
        # Generate CAM
        cam = self.gradcam.generate_cam(input_tensor, target_class)
        
        # Visualize
        vis = self.gradcam.visualize(img, cam, output_path)
        
        return {
            "predicted_class": pred_class,
            "confidence": float(confidence),
            "target_class": target_class or pred_class,
            "cam_shape": cam.shape,
            "visualization": vis if output_path is None else output_path
        }


def main():
    """Demo Grad-CAM"""
    import argparse
    from main import build_convnext_classifier, ClassificationTransform
    
    parser = argparse.ArgumentParser(description="Generate Grad-CAM explanations")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Input image")
    parser.add_argument("--output", type=str, default="gradcam_output.png", help="Output path")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    
    args = parser.parse_args()
    
    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    variant = checkpoint["variant"]
    num_classes = checkpoint["num_classes"]
    
    model = build_convnext_classifier(variant, num_classes, dropout=0.0)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
    # Initialize explainability engine
    engine = ExplainabilityEngine(model, args.device)
    
    # Create preprocessing function
    transform = ClassificationTransform(size=args.img_size, train=False)
    
    # Generate explanation
    result = engine.explain_prediction(
        args.image,
        transform,
        output_path=args.output
    )
    
    print(f"\n[Results]")
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Explanation saved to: {args.output}")


if __name__ == "__main__":
    main()
