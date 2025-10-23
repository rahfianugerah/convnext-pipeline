#!/usr/bin/env python3
"""
Visualization tools for predictions
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, Any, List, Optional
import json


class PredictionVisualizer:
    """Visualize model predictions"""
    
    # Color palette for bounding boxes
    COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128)
    ]
    
    @staticmethod
    def visualize_classification(image_path: str, prediction: Dict[str, Any],
                                 output_path: Optional[str] = None,
                                 top_k: int = 5) -> np.ndarray:
        """
        Visualize classification predictions
        
        Args:
            image_path: Path to input image
            prediction: Prediction dictionary
            output_path: Optional path to save visualization
            top_k: Number of top predictions to show
            
        Returns:
            Visualization as numpy array
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Show image
        ax1.imshow(img)
        ax1.axis('off')
        ax1.set_title(f"Prediction: {prediction.get('class_name', 'Unknown')}\n"
                     f"Confidence: {prediction.get('confidence', 0):.2%}")
        
        # Show top-k probabilities
        probs = prediction.get('probabilities', {})
        if probs:
            # Sort by probability
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:top_k]
            classes = [item[0] for item in sorted_probs]
            values = [item[1] for item in sorted_probs]
            
            ax2.barh(classes, values)
            ax2.set_xlabel('Probability')
            ax2.set_title(f'Top-{top_k} Predictions')
            ax2.set_xlim(0, 1)
        
        plt.tight_layout()
        
        # Save or show
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"[INFO] Visualization saved to {output_path}")
        
        # Convert to numpy array
        fig.canvas.draw()
        vis = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis = vis.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return vis
    
    @staticmethod
    def visualize_detection(image_path: str, prediction: Dict[str, Any],
                           output_path: Optional[str] = None,
                           conf_threshold: float = 0.5) -> np.ndarray:
        """
        Visualize object detection predictions
        
        Args:
            image_path: Path to input image
            prediction: Prediction dictionary with detections
            output_path: Optional path to save visualization
            conf_threshold: Confidence threshold for display
            
        Returns:
            Visualization as numpy array
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img)
        
        # Draw detections
        detections = prediction.get('detections', [])
        for i, det in enumerate(detections):
            if det['confidence'] < conf_threshold:
                continue
            
            # Get bbox coordinates
            bbox = det['bbox']  # [xmin, ymin, xmax, ymax]
            x1, y1, x2, y2 = bbox
            
            # Get color
            color_idx = det['label'] % len(PredictionVisualizer.COLORS)
            color = tuple(c / 255.0 for c in PredictionVisualizer.COLORS[color_idx])
            
            # Draw rectangle
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            ax.text(
                x1, y1 - 5,
                label,
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7),
                fontsize=10,
                color='white'
            )
        
        ax.axis('off')
        ax.set_title(f"Detected {len(detections)} objects")
        
        plt.tight_layout()
        
        # Save or show
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"[INFO] Visualization saved to {output_path}")
        
        # Convert to numpy array
        fig.canvas.draw()
        vis = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis = vis.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return vis
    
    @staticmethod
    def visualize_ocr(image_path: str, prediction: Dict[str, Any],
                     output_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize OCR predictions
        
        Args:
            image_path: Path to input image
            prediction: Prediction dictionary with text
            output_path: Optional path to save visualization
            
        Returns:
            Visualization as numpy array
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.imshow(img)
        ax.axis('off')
        
        # Add recognized text
        text = prediction.get('text', '')
        confidence = prediction.get('confidence', 0)
        
        title = f"Recognized Text: '{text}'\nConfidence: {confidence:.2%}"
        ax.set_title(title, fontsize=14, pad=20)
        
        plt.tight_layout()
        
        # Save or show
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"[INFO] Visualization saved to {output_path}")
        
        # Convert to numpy array
        fig.canvas.draw()
        vis = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis = vis.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return vis
    
    @staticmethod
    def visualize_batch(predictions_file: str, images_dir: str, 
                       output_dir: str, max_images: int = 10):
        """
        Visualize batch of predictions
        
        Args:
            predictions_file: Path to predictions JSON file
            images_dir: Directory containing images
            output_dir: Directory to save visualizations
            max_images: Maximum number of images to visualize
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load predictions
        with open(predictions_file, 'r') as f:
            predictions = json.load(f)
        
        if not isinstance(predictions, list):
            predictions = [predictions]
        
        # Visualize each prediction
        for i, pred in enumerate(predictions[:max_images]):
            if 'image_path' not in pred:
                continue
            
            image_path = pred['image_path']
            theme = pred.get('theme', 'unknown')
            
            # Generate output filename
            img_name = Path(image_path).stem
            output_file = output_path / f"{img_name}_vis.png"
            
            try:
                if theme == 'classification':
                    PredictionVisualizer.visualize_classification(
                        image_path, pred, str(output_file)
                    )
                elif theme == 'object_detection' or theme == 'object':
                    PredictionVisualizer.visualize_detection(
                        image_path, pred, str(output_file)
                    )
                elif theme == 'ocr':
                    PredictionVisualizer.visualize_ocr(
                        image_path, pred, str(output_file)
                    )
                
                print(f"[{i+1}/{len(predictions)}] Processed {image_path}")
                
            except Exception as e:
                print(f"[ERROR] Failed to visualize {image_path}: {str(e)}")
        
        print(f"\n[INFO] Visualizations saved to {output_dir}")


def main():
    """Demo visualization"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize predictions")
    parser.add_argument("--predictions", type=str, required=True, help="Predictions JSON file")
    parser.add_argument("--images_dir", type=str, default=".", help="Images directory")
    parser.add_argument("--output_dir", type=str, default="./visualizations", help="Output directory")
    parser.add_argument("--max_images", type=int, default=10, help="Max images to visualize")
    
    args = parser.parse_args()
    
    PredictionVisualizer.visualize_batch(
        args.predictions,
        args.images_dir,
        args.output_dir,
        args.max_images
    )


if __name__ == "__main__":
    main()
