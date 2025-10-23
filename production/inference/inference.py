#!/usr/bin/env python3
"""
Production Inference Script for ConvNeXt Multi-Task Pipeline
Supports classification, object detection, and OCR inference
"""

import os
import cv2
import json
import torch
import timm
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Union, Tuple
from PIL import Image
import torch.nn as nn
from torchvision.transforms import functional as TF
import torchvision

# Import model architectures from main.py
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from main import (
    build_convnext_classifier,
    build_faster_rcnn_with_convnext,
    OCRConvNeXtCTC,
    OCRLabelEncoder,
    ClassificationTransform,
    ObjectDetectionTransform,
    OCRTransform
)


class InferenceEngine:
    """Production inference engine for all three tasks"""
    
    def __init__(self, checkpoint_path: str, theme: str, device: str = "auto"):
        """
        Initialize inference engine
        
        Args:
            checkpoint_path: Path to model checkpoint
            theme: Task type ('classification', 'object', 'ocr')
            device: Device to run inference on ('cuda', 'cpu', or 'auto')
        """
        self.theme = theme
        self.checkpoint_path = Path(checkpoint_path)
        
        # Auto-detect device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model, self.metadata = self._load_model()
        self.model.eval()
        
        print(f"[INFO] Loaded {theme} model on {self.device}")
        print(f"[INFO] Model metadata: {self.metadata}")
    
    def _load_model(self) -> Tuple[nn.Module, Dict]:
        """Load model checkpoint and metadata"""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        metadata = {k: v for k, v in checkpoint.items() if k != "model"}
        
        # Build model based on theme
        if self.theme == "classification":
            variant = checkpoint["variant"]
            num_classes = checkpoint["num_classes"]
            model = build_convnext_classifier(variant, num_classes, dropout=0.0)
            model.load_state_dict(checkpoint["model"])
            
        elif self.theme == "object":
            variant = checkpoint["variant"]
            classes = checkpoint["classes"]
            num_classes = len(classes) + 1  # +1 for background
            model = build_faster_rcnn_with_convnext(variant, num_classes=num_classes)
            model.load_state_dict(checkpoint["model"])
            metadata["class_names"] = classes
            
        elif self.theme == "ocr":
            variant = checkpoint["variant"]
            charset = checkpoint.get("charset", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
            encoder = OCRLabelEncoder(charset)
            vocab_size = encoder.vocab_size
            model = OCRConvNeXtCTC(variant=variant, vocab_size=vocab_size)
            model.load_state_dict(checkpoint["model"])
            metadata["encoder"] = encoder
            metadata["charset"] = charset
            
        else:
            raise ValueError(f"Unknown theme: {self.theme}")
        
        return model.to(self.device), metadata
    
    @torch.no_grad()
    def predict(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """
        Run inference on a single image
        
        Args:
            image_path: Path to input image
            **kwargs: Additional parameters for inference
            
        Returns:
            Dictionary with predictions
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Read image
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        # Run theme-specific prediction
        if self.theme == "classification":
            return self._predict_classification(img, **kwargs)
        elif self.theme == "object":
            return self._predict_object_detection(img, **kwargs)
        elif self.theme == "ocr":
            return self._predict_ocr(img, **kwargs)
        else:
            raise ValueError(f"Unknown theme: {self.theme}")
    
    def _predict_classification(self, img: np.ndarray, img_size: int = 224) -> Dict[str, Any]:
        """Classification inference"""
        # Transform
        transform = ClassificationTransform(size=img_size, train=False)
        x = transform(img).unsqueeze(0).to(self.device)
        
        # Inference
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        pred_class = logits.argmax(1).item()
        confidence = probs[0, pred_class].item()
        
        # Get class names if available
        num_classes = self.metadata.get("num_classes", logits.shape[1])
        class_names = [f"class_{i}" for i in range(num_classes)]
        
        return {
            "predicted_class": pred_class,
            "class_name": class_names[pred_class],
            "confidence": float(confidence),
            "probabilities": {class_names[i]: float(probs[0, i].item()) 
                            for i in range(num_classes)},
            "theme": "classification"
        }
    
    def _predict_object_detection(self, img: np.ndarray, det_size: int = 640, 
                                   conf_threshold: float = 0.5) -> Dict[str, Any]:
        """Object detection inference"""
        # Transform
        transform = ObjectDetectionTransform(size=det_size, train=False)
        boxes_dummy = np.zeros((0, 4))
        x, _ = transform(img, boxes_dummy)
        x = x.unsqueeze(0).to(self.device)
        
        # Inference
        predictions = self.model(x)[0]
        
        # Filter by confidence
        keep = predictions["scores"] > conf_threshold
        boxes = predictions["boxes"][keep].cpu().numpy()
        labels = predictions["labels"][keep].cpu().numpy()
        scores = predictions["scores"][keep].cpu().numpy()
        
        # Get class names
        class_names = self.metadata.get("class_names", [])
        
        detections = []
        for box, label, score in zip(boxes, labels, scores):
            det = {
                "bbox": [float(x) for x in box],  # [xmin, ymin, xmax, ymax]
                "label": int(label),
                "class_name": class_names[label-1] if label > 0 and label-1 < len(class_names) else f"class_{label}",
                "confidence": float(score)
            }
            detections.append(det)
        
        return {
            "detections": detections,
            "num_objects": len(detections),
            "theme": "object_detection"
        }
    
    def _predict_ocr(self, img: np.ndarray, ocr_h: int = 32, ocr_w: int = 256) -> Dict[str, Any]:
        """OCR inference"""
        # Transform
        transform = OCRTransform(img_h=ocr_h, img_w=ocr_w, train=False)
        x = transform(img).unsqueeze(0).to(self.device)
        
        # Inference
        logits = self.model(x)  # T x B x V
        
        # Decode
        encoder = self.metadata.get("encoder")
        if encoder is None:
            raise ValueError("OCR encoder not found in metadata")
        
        predicted_texts = encoder.decode_greedy(logits)
        text = predicted_texts[0] if predicted_texts else ""
        
        # Get confidence (average of max probs per timestep)
        probs = logits.softmax(-1)
        max_probs = probs.max(-1)[0]  # T x B
        avg_confidence = max_probs.mean().item()
        
        return {
            "text": text,
            "confidence": float(avg_confidence),
            "theme": "ocr"
        }
    
    def predict_batch(self, image_paths: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Run inference on multiple images
        
        Args:
            image_paths: List of image paths
            **kwargs: Additional parameters
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for img_path in image_paths:
            try:
                result = self.predict(img_path, **kwargs)
                result["image_path"] = img_path
                results.append(result)
            except Exception as e:
                results.append({
                    "image_path": img_path,
                    "error": str(e),
                    "theme": self.theme
                })
        return results


def main():
    parser = argparse.ArgumentParser(description="Production Inference for ConvNeXt Models")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--theme", type=str, required=True, choices=["classification", "object", "ocr"])
    parser.add_argument("--image", type=str, help="Single image path")
    parser.add_argument("--images", type=str, nargs="+", help="Multiple image paths")
    parser.add_argument("--image_dir", type=str, help="Directory of images")
    parser.add_argument("--output", type=str, default="predictions.json", help="Output JSON file")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--img_size", type=int, default=224, help="Image size for classification")
    parser.add_argument("--det_size", type=int, default=640, help="Image size for detection")
    parser.add_argument("--ocr_h", type=int, default=32, help="OCR image height")
    parser.add_argument("--ocr_w", type=int, default=256, help="OCR image width")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="Confidence threshold for detection")
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = InferenceEngine(args.checkpoint, args.theme, args.device)
    
    # Collect images
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    if args.images:
        image_paths.extend(args.images)
    if args.image_dir:
        img_dir = Path(args.image_dir)
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
            image_paths.extend([str(p) for p in img_dir.glob(ext)])
    
    if not image_paths:
        print("[ERROR] No images provided. Use --image, --images, or --image_dir")
        return
    
    print(f"[INFO] Processing {len(image_paths)} images...")
    
    # Run inference
    kwargs = {
        "img_size": args.img_size,
        "det_size": args.det_size,
        "ocr_h": args.ocr_h,
        "ocr_w": args.ocr_w,
        "conf_threshold": args.conf_threshold
    }
    
    if len(image_paths) == 1:
        result = engine.predict(image_paths[0], **kwargs)
        result["image_path"] = image_paths[0]
        results = [result]
    else:
        results = engine.predict_batch(image_paths, **kwargs)
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"[INFO] Results saved to {args.output}")
    print(f"[INFO] Processed {len(results)} images")
    
    # Print summary
    for i, res in enumerate(results[:3]):  # Show first 3
        print(f"\n[Result {i+1}] {res.get('image_path', 'unknown')}")
        if "error" in res:
            print(f"  Error: {res['error']}")
        elif args.theme == "classification":
            print(f"  Class: {res['class_name']} (confidence: {res['confidence']:.3f})")
        elif args.theme == "object":
            print(f"  Objects detected: {res['num_objects']}")
        elif args.theme == "ocr":
            print(f"  Text: {res['text']} (confidence: {res['confidence']:.3f})")
    
    if len(results) > 3:
        print(f"\n... and {len(results) - 3} more results")


if __name__ == "__main__":
    main()
