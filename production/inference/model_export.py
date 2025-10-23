#!/usr/bin/env python3
"""
Model Export to ONNX and TorchScript for production deployment
"""

import torch
import torch.nn as nn
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from main import (
    build_convnext_classifier,
    build_faster_rcnn_with_convnext,
    OCRConvNeXtCTC
)


class ModelExporter:
    """Export trained models to production formats"""
    
    def __init__(self, checkpoint_path: str, theme: str):
        """
        Initialize exporter
        
        Args:
            checkpoint_path: Path to PyTorch checkpoint
            theme: Task type ('classification', 'object', 'ocr')
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.theme = theme
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load checkpoint
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model = self._load_model()
        self.model.eval()
    
    def _load_model(self) -> nn.Module:
        """Load model from checkpoint"""
        if self.theme == "classification":
            variant = self.checkpoint["variant"]
            num_classes = self.checkpoint["num_classes"]
            model = build_convnext_classifier(variant, num_classes, dropout=0.0)
            model.load_state_dict(self.checkpoint["model"])
            
        elif self.theme == "object":
            variant = self.checkpoint["variant"]
            classes = self.checkpoint["classes"]
            num_classes = len(classes) + 1
            model = build_faster_rcnn_with_convnext(variant, num_classes=num_classes)
            model.load_state_dict(self.checkpoint["model"])
            
        elif self.theme == "ocr":
            variant = self.checkpoint["variant"]
            charset = self.checkpoint.get("charset", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
            from main import OCRLabelEncoder
            encoder = OCRLabelEncoder(charset)
            vocab_size = encoder.vocab_size
            model = OCRConvNeXtCTC(variant=variant, vocab_size=vocab_size)
            model.load_state_dict(self.checkpoint["model"])
            
        else:
            raise ValueError(f"Unknown theme: {self.theme}")
        
        return model.to(self.device)
    
    def export_torchscript(self, output_path: str, method: str = "trace") -> Dict[str, Any]:
        """
        Export model to TorchScript format
        
        Args:
            output_path: Output file path (.pt)
            method: Export method ('trace' or 'script')
            
        Returns:
            Export metadata
        """
        print(f"[INFO] Exporting {self.theme} model to TorchScript using {method}...")
        
        # Create example input based on theme
        if self.theme == "classification":
            example_input = torch.randn(1, 3, 224, 224).to(self.device)
        elif self.theme == "object":
            example_input = torch.randn(1, 3, 640, 640).to(self.device)
        elif self.theme == "ocr":
            example_input = torch.randn(1, 1, 32, 256).to(self.device)
        else:
            raise ValueError(f"Unknown theme: {self.theme}")
        
        try:
            # Export based on method
            if method == "trace":
                traced_model = torch.jit.trace(self.model, example_input)
            elif method == "script":
                traced_model = torch.jit.script(self.model)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Save
            traced_model.save(output_path)
            
            # Get file size
            file_size = Path(output_path).stat().st_size / 1024 / 1024  # MB
            
            print(f"[SUCCESS] TorchScript model saved to {output_path}")
            print(f"[INFO] Model size: {file_size:.2f} MB")
            
            # Save metadata
            metadata = {
                "format": "torchscript",
                "method": method,
                "theme": self.theme,
                "variant": self.checkpoint.get("variant"),
                "file_size_mb": file_size,
                "output_path": str(output_path)
            }
            
            # Add theme-specific metadata
            if self.theme == "classification":
                metadata["num_classes"] = self.checkpoint.get("num_classes")
            elif self.theme == "object":
                metadata["classes"] = self.checkpoint.get("classes")
            elif self.theme == "ocr":
                metadata["charset"] = self.checkpoint.get("charset")
            
            # Save metadata
            metadata_path = Path(output_path).with_suffix(".json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            return metadata
            
        except Exception as e:
            print(f"[ERROR] TorchScript export failed: {str(e)}")
            raise
    
    def export_onnx(self, output_path: str, opset_version: int = 14,
                   dynamic_axes: bool = True) -> Dict[str, Any]:
        """
        Export model to ONNX format
        
        Args:
            output_path: Output file path (.onnx)
            opset_version: ONNX opset version
            dynamic_axes: Whether to use dynamic batch size
            
        Returns:
            Export metadata
        """
        print(f"[INFO] Exporting {self.theme} model to ONNX (opset {opset_version})...")
        
        # Create example input and configure dynamic axes
        if self.theme == "classification":
            example_input = torch.randn(1, 3, 224, 224).to(self.device)
            input_names = ["image"]
            output_names = ["logits"]
            dynamic_axes_dict = {
                "image": {0: "batch_size"},
                "logits": {0: "batch_size"}
            } if dynamic_axes else None
            
        elif self.theme == "object":
            # Note: Faster R-CNN export to ONNX is complex and may not work perfectly
            example_input = torch.randn(1, 3, 640, 640).to(self.device)
            input_names = ["image"]
            output_names = ["boxes", "labels", "scores"]
            dynamic_axes_dict = {
                "image": {0: "batch_size"}
            } if dynamic_axes else None
            print("[WARNING] Object detection models may have limited ONNX compatibility")
            
        elif self.theme == "ocr":
            example_input = torch.randn(1, 1, 32, 256).to(self.device)
            input_names = ["image"]
            output_names = ["logits"]
            dynamic_axes_dict = {
                "image": {0: "batch_size"},
                "logits": {1: "batch_size"}  # T x B x V
            } if dynamic_axes else None
            
        else:
            raise ValueError(f"Unknown theme: {self.theme}")
        
        try:
            # Export to ONNX
            torch.onnx.export(
                self.model,
                example_input,
                output_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes_dict,
                opset_version=opset_version,
                do_constant_folding=True,
                export_params=True
            )
            
            # Get file size
            file_size = Path(output_path).stat().st_size / 1024 / 1024  # MB
            
            print(f"[SUCCESS] ONNX model saved to {output_path}")
            print(f"[INFO] Model size: {file_size:.2f} MB")
            
            # Verify ONNX model
            try:
                import onnx
                onnx_model = onnx.load(output_path)
                onnx.checker.check_model(onnx_model)
                print("[INFO] ONNX model verification passed")
            except ImportError:
                print("[WARNING] onnx package not installed, skipping verification")
            except Exception as e:
                print(f"[WARNING] ONNX verification failed: {str(e)}")
            
            # Save metadata
            metadata = {
                "format": "onnx",
                "opset_version": opset_version,
                "dynamic_axes": dynamic_axes,
                "theme": self.theme,
                "variant": self.checkpoint.get("variant"),
                "file_size_mb": file_size,
                "input_names": input_names,
                "output_names": output_names,
                "output_path": str(output_path)
            }
            
            # Add theme-specific metadata
            if self.theme == "classification":
                metadata["num_classes"] = self.checkpoint.get("num_classes")
                metadata["input_shape"] = [1, 3, 224, 224]
            elif self.theme == "object":
                metadata["classes"] = self.checkpoint.get("classes")
                metadata["input_shape"] = [1, 3, 640, 640]
            elif self.theme == "ocr":
                metadata["charset"] = self.checkpoint.get("charset")
                metadata["input_shape"] = [1, 1, 32, 256]
            
            # Save metadata
            metadata_path = Path(output_path).with_suffix(".json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            return metadata
            
        except Exception as e:
            print(f"[ERROR] ONNX export failed: {str(e)}")
            if self.theme == "object":
                print("[INFO] Faster R-CNN models have limited ONNX support. Consider using TorchScript instead.")
            raise
    
    def export_all(self, output_dir: str) -> Dict[str, Any]:
        """
        Export model to all available formats
        
        Args:
            output_dir: Output directory
            
        Returns:
            Combined metadata
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            "theme": self.theme,
            "source_checkpoint": str(self.checkpoint_path),
            "exports": {}
        }
        
        # Export TorchScript
        try:
            ts_path = output_dir / f"{self.theme}_model.pt"
            ts_metadata = self.export_torchscript(str(ts_path), method="trace")
            results["exports"]["torchscript"] = ts_metadata
        except Exception as e:
            print(f"[ERROR] TorchScript export failed: {str(e)}")
            results["exports"]["torchscript"] = {"error": str(e)}
        
        # Export ONNX (skip for object detection due to compatibility issues)
        if self.theme != "object":
            try:
                onnx_path = output_dir / f"{self.theme}_model.onnx"
                onnx_metadata = self.export_onnx(str(onnx_path))
                results["exports"]["onnx"] = onnx_metadata
            except Exception as e:
                print(f"[ERROR] ONNX export failed: {str(e)}")
                results["exports"]["onnx"] = {"error": str(e)}
        else:
            print("[INFO] Skipping ONNX export for object detection (limited compatibility)")
            results["exports"]["onnx"] = {"skipped": "limited compatibility for object detection"}
        
        # Save combined metadata
        summary_path = output_dir / "export_summary.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n[SUCCESS] Export summary saved to {summary_path}")
        return results


def main():
    parser = argparse.ArgumentParser(description="Export models to production formats")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to PyTorch checkpoint")
    parser.add_argument("--theme", type=str, required=True, choices=["classification", "object", "ocr"])
    parser.add_argument("--output_dir", type=str, default="./exported_models", help="Output directory")
    parser.add_argument("--format", type=str, default="all", 
                       choices=["torchscript", "onnx", "all"], help="Export format")
    parser.add_argument("--opset_version", type=int, default=14, help="ONNX opset version")
    
    args = parser.parse_args()
    
    # Initialize exporter
    exporter = ModelExporter(args.checkpoint, args.theme)
    
    # Export based on format
    if args.format == "all":
        exporter.export_all(args.output_dir)
    elif args.format == "torchscript":
        output_path = Path(args.output_dir) / f"{args.theme}_model.pt"
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        exporter.export_torchscript(str(output_path))
    elif args.format == "onnx":
        output_path = Path(args.output_dir) / f"{args.theme}_model.onnx"
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        exporter.export_onnx(str(output_path), opset_version=args.opset_version)


if __name__ == "__main__":
    main()
