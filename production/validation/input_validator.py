#!/usr/bin/env python3
"""
Input validation and data preprocessing for production
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image
import mimetypes


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


class InputValidator:
    """Validates and preprocesses input data for production inference"""
    
    # Supported image formats
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
    
    # Image size constraints
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    MIN_DIMENSION = 16
    MAX_DIMENSION = 4096
    
    def __init__(self, theme: str, strict: bool = True):
        """
        Initialize validator
        
        Args:
            theme: Task type ('classification', 'object', 'ocr')
            strict: If True, raise errors; if False, return warnings
        """
        self.theme = theme
        self.strict = strict
    
    def validate_image_path(self, image_path: str) -> Dict[str, Any]:
        """
        Validate image file path and format
        
        Returns:
            Dictionary with validation results
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "path": image_path
        }
        
        path = Path(image_path)
        
        # Check if file exists
        if not path.exists():
            result["valid"] = False
            result["errors"].append(f"File not found: {image_path}")
            if self.strict:
                raise ValidationError(f"File not found: {image_path}")
            return result
        
        # Check if it's a file
        if not path.is_file():
            result["valid"] = False
            result["errors"].append(f"Path is not a file: {image_path}")
            if self.strict:
                raise ValidationError(f"Path is not a file: {image_path}")
            return result
        
        # Check file extension
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            result["valid"] = False
            result["errors"].append(
                f"Unsupported format: {path.suffix}. "
                f"Supported: {', '.join(self.SUPPORTED_FORMATS)}"
            )
            if self.strict:
                raise ValidationError(f"Unsupported format: {path.suffix}")
            return result
        
        # Check file size
        file_size = path.stat().st_size
        if file_size > self.MAX_IMAGE_SIZE:
            result["valid"] = False
            result["errors"].append(
                f"File too large: {file_size / 1024 / 1024:.2f}MB "
                f"(max: {self.MAX_IMAGE_SIZE / 1024 / 1024:.0f}MB)"
            )
            if self.strict:
                raise ValidationError("File too large")
            return result
        
        if file_size == 0:
            result["valid"] = False
            result["errors"].append("File is empty")
            if self.strict:
                raise ValidationError("File is empty")
            return result
        
        return result
    
    def validate_image_content(self, image_path: str) -> Dict[str, Any]:
        """
        Validate image content (can be loaded, has valid dimensions, etc.)
        
        Returns:
            Dictionary with validation results and image metadata
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "metadata": {}
        }
        
        try:
            # Try to load with OpenCV
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                # Try with PIL as fallback
                try:
                    pil_img = Image.open(image_path)
                    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    result["valid"] = False
                    result["errors"].append(f"Failed to load image: {str(e)}")
                    if self.strict:
                        raise ValidationError(f"Failed to load image: {str(e)}")
                    return result
            
            # Get dimensions
            height, width = img.shape[:2]
            channels = img.shape[2] if len(img.shape) == 3 else 1
            
            result["metadata"] = {
                "height": height,
                "width": width,
                "channels": channels,
                "dtype": str(img.dtype)
            }
            
            # Validate dimensions
            if height < self.MIN_DIMENSION or width < self.MIN_DIMENSION:
                result["valid"] = False
                result["errors"].append(
                    f"Image too small: {width}x{height}. "
                    f"Minimum: {self.MIN_DIMENSION}x{self.MIN_DIMENSION}"
                )
                if self.strict:
                    raise ValidationError("Image too small")
            
            if height > self.MAX_DIMENSION or width > self.MAX_DIMENSION:
                result["warnings"].append(
                    f"Very large image: {width}x{height}. "
                    f"This may cause memory issues."
                )
            
            # Task-specific validation
            if self.theme == "ocr":
                aspect_ratio = width / height
                if aspect_ratio < 1.0:
                    result["warnings"].append(
                        f"OCR works best with horizontal text (aspect ratio {aspect_ratio:.2f} < 1.0)"
                    )
                if height < 24:
                    result["warnings"].append(
                        f"Image height ({height}px) is quite small for OCR. Text may not be recognized well."
                    )
            
            if self.theme == "classification":
                if width < 64 or height < 64:
                    result["warnings"].append(
                        "Small image for classification. Quality may affect accuracy."
                    )
            
            # Check if image is corrupted (all zeros or constant)
            if img.std() < 1.0:
                result["warnings"].append("Image appears to have very low variance (possibly corrupted)")
            
        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"Validation error: {str(e)}")
            if self.strict:
                raise ValidationError(f"Validation error: {str(e)}")
        
        return result
    
    def validate_and_preprocess(self, image_path: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Complete validation pipeline
        
        Returns:
            (is_valid, validation_report)
        """
        report = {
            "path": image_path,
            "theme": self.theme,
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Validate path
        path_result = self.validate_image_path(image_path)
        if not path_result["valid"]:
            report["valid"] = False
            report["errors"].extend(path_result["errors"])
            return False, report
        
        report["warnings"].extend(path_result["warnings"])
        
        # Validate content
        content_result = self.validate_image_content(image_path)
        if not content_result["valid"]:
            report["valid"] = False
            report["errors"].extend(content_result["errors"])
            return False, report
        
        report["warnings"].extend(content_result["warnings"])
        report["metadata"] = content_result.get("metadata", {})
        
        return True, report
    
    def validate_batch(self, image_paths: List[str]) -> Dict[str, Any]:
        """
        Validate multiple images
        
        Returns:
            Summary report for all images
        """
        results = {
            "total": len(image_paths),
            "valid": 0,
            "invalid": 0,
            "warnings": 0,
            "details": []
        }
        
        for img_path in image_paths:
            is_valid, report = self.validate_and_preprocess(img_path)
            results["details"].append(report)
            
            if is_valid:
                results["valid"] += 1
            else:
                results["invalid"] += 1
            
            if report["warnings"]:
                results["warnings"] += len(report["warnings"])
        
        return results


class DataPreprocessor:
    """Handles data preprocessing for production"""
    
    @staticmethod
    def auto_orient(img: np.ndarray) -> np.ndarray:
        """Auto-orient image based on EXIF data (if available)"""
        # This is a simplified version - full implementation would use PIL EXIF
        return img
    
    @staticmethod
    def normalize_brightness(img: np.ndarray, target_mean: float = 127.5) -> np.ndarray:
        """Normalize image brightness"""
        current_mean = img.mean()
        if current_mean > 0:
            scale = target_mean / current_mean
            img = np.clip(img * scale, 0, 255).astype(np.uint8)
        return img
    
    @staticmethod
    def remove_noise(img: np.ndarray) -> np.ndarray:
        """Apply light denoising"""
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
    @staticmethod
    def enhance_contrast(img: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    @staticmethod
    def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
        """Specialized preprocessing for OCR"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while keeping edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        return binary


def main():
    """Demo validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate input images")
    parser.add_argument("--theme", type=str, required=True, choices=["classification", "object", "ocr"])
    parser.add_argument("--image", type=str, help="Single image to validate")
    parser.add_argument("--images", type=str, nargs="+", help="Multiple images")
    parser.add_argument("--strict", action="store_true", help="Strict mode (raise errors)")
    
    args = parser.parse_args()
    
    validator = InputValidator(args.theme, strict=args.strict)
    
    if args.image:
        is_valid, report = validator.validate_and_preprocess(args.image)
        print(f"\n[Validation Result]")
        print(f"Valid: {is_valid}")
        if report["errors"]:
            print(f"Errors: {report['errors']}")
        if report["warnings"]:
            print(f"Warnings: {report['warnings']}")
        if "metadata" in report:
            print(f"Metadata: {report['metadata']}")
    
    elif args.images:
        results = validator.validate_batch(args.images)
        print(f"\n[Batch Validation Results]")
        print(f"Total: {results['total']}")
        print(f"Valid: {results['valid']}")
        print(f"Invalid: {results['invalid']}")
        print(f"Total warnings: {results['warnings']}")


if __name__ == "__main__":
    main()
