"""
Unit tests for production components
"""

import pytest
import json
import tempfile
import numpy as np
import cv2
from pathlib import Path


# Test Input Validator
def test_input_validator():
    """Test input validation"""
    from production.validation.input_validator import InputValidator
    
    validator = InputValidator("classification", strict=False)
    
    # Create a temporary test image
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        cv2.imwrite(tmp.name, img)
        tmp_path = tmp.name
    
    try:
        # Test path validation
        result = validator.validate_image_path(tmp_path)
        assert result["valid"] == True
        assert len(result["errors"]) == 0
        
        # Test content validation
        result = validator.validate_image_content(tmp_path)
        assert result["valid"] == True
        assert "metadata" in result
        assert result["metadata"]["height"] == 224
        assert result["metadata"]["width"] == 224
        
        # Test complete validation
        is_valid, report = validator.validate_and_preprocess(tmp_path)
        assert is_valid == True
    finally:
        Path(tmp_path).unlink()


def test_input_validator_invalid_path():
    """Test validation with invalid path"""
    from production.validation.input_validator import InputValidator
    
    validator = InputValidator("classification", strict=False)
    
    result = validator.validate_image_path("nonexistent.jpg")
    assert result["valid"] == False
    assert len(result["errors"]) > 0


# Test Prediction Logger
def test_prediction_logger():
    """Test prediction logging"""
    from production.monitoring.logger import PredictionLogger
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = PredictionLogger(tmpdir)
        
        # Log some predictions
        for i in range(5):
            logger.log_prediction(
                theme="classification",
                prediction={"class": i, "confidence": 0.95},
                metadata={"inference_time_ms": 50}
            )
        
        # Get metrics
        metrics = logger.get_metrics()
        assert metrics["total_predictions"] == 5
        assert metrics["predictions_by_theme"]["classification"] == 5
        assert metrics["avg_inference_time_ms"] == 50
        
        # Test error logging
        logger.log_error("classification", "Test error")
        metrics = logger.get_metrics()
        assert metrics["errors"] == 1
        
        logger.close()


# Test Model Registry
def test_model_registry():
    """Test model versioning"""
    from production.versioning.registry import ModelRegistry
    
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ModelRegistry(tmpdir)
        
        # Create a dummy model file
        model_path = Path(tmpdir) / "test_model.pth"
        model_path.write_text("dummy model data")
        
        # Register model
        version = registry.register_model(
            str(model_path),
            theme="classification",
            metadata={"accuracy": 0.95}
        )
        
        assert version.version == "v1"
        assert version.metadata["accuracy"] == 0.95
        
        # Get model
        retrieved = registry.get_model("classification", "v1")
        assert retrieved is not None
        assert retrieved.version == "v1"
        
        # Get latest
        latest = registry.get_latest_model("classification")
        assert latest is not None
        assert latest.version == "v1"
        
        # List models
        models = registry.list_models("classification")
        assert len(models) == 1
        
        # Tag version
        registry.tag_version("classification", "v1", ["production"])
        
        # Promote version
        registry.promote_version("classification", "v1", "production")
        
        # Get production model
        prod_model = registry.get_production_model("classification")
        assert prod_model is not None
        assert prod_model.version == "v1"


# Test A/B Testing
def test_ab_testing():
    """Test A/B testing manager"""
    from production.ab_testing.manager import ABTestManager
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "ab_config.json"
        manager = ABTestManager(str(config_path))
        
        # Create experiment
        exp = manager.create_experiment(
            name="test_exp",
            model_a_path="/path/to/model_a",
            model_b_path="/path/to/model_b",
            split=0.5
        )
        
        assert exp["name"] == "test_exp"
        assert exp["split"] == 0.5
        
        # Assign variants
        variant = manager.assign_variant("test_exp", "user123")
        assert variant in ["A", "B"]
        
        # Record results
        for i in range(10):
            manager.record_result(
                "test_exp",
                "A",
                {"accuracy": 0.9 + i * 0.01}
            )
            manager.record_result(
                "test_exp",
                "B",
                {"accuracy": 0.92 + i * 0.01}
            )
        
        # Analyze
        analysis = manager.analyze_experiment("test_exp")
        assert "variant_a" in analysis
        assert "variant_b" in analysis
        assert analysis["variant_a"]["samples"] == 10
        assert analysis["variant_b"]["samples"] == 10


# Test Visualization
def test_visualization():
    """Test prediction visualization"""
    from production.utils.visualization import PredictionVisualizer
    
    # Create test image
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        cv2.imwrite(tmp.name, img)
        tmp_path = tmp.name
    
    try:
        # Test classification visualization
        prediction = {
            "class_name": "test_class",
            "confidence": 0.95,
            "probabilities": {
                "class_0": 0.95,
                "class_1": 0.03,
                "class_2": 0.02
            }
        }
        
        vis = PredictionVisualizer.visualize_classification(
            tmp_path,
            prediction,
            output_path=None
        )
        
        assert vis is not None
        assert isinstance(vis, np.ndarray)
        assert len(vis.shape) == 3
        
    finally:
        Path(tmp_path).unlink()


# Test Data Preprocessing
def test_data_preprocessing():
    """Test data preprocessing utilities"""
    from production.validation.input_validator import DataPreprocessor
    
    # Create test image
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Test contrast enhancement
    enhanced = DataPreprocessor.enhance_contrast(img)
    assert enhanced.shape == img.shape
    
    # Test brightness normalization
    normalized = DataPreprocessor.normalize_brightness(img)
    assert normalized.shape == img.shape


# Test API endpoints (requires running server)
@pytest.mark.skip(reason="Requires running server")
def test_api_health():
    """Test API health endpoint"""
    import requests
    
    response = requests.get("http://localhost:8000/health")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


@pytest.mark.skip(reason="Requires running server")
def test_api_predict():
    """Test API prediction endpoint"""
    import requests
    
    # Create test image
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        cv2.imwrite(tmp.name, img)
        tmp_path = tmp.name
    
    try:
        with open(tmp_path, 'rb') as f:
            response = requests.post(
                "http://localhost:8000/predict",
                files={"file": f}
            )
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == True
        assert "prediction" in data
        
    finally:
        Path(tmp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
