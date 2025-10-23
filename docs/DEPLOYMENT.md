# Production Deployment Guide

## Overview

This guide covers deploying the ConvNeXt Multi-Task Pipeline to production using Docker, the REST API, and various production-ready components.

## Architecture

```
┌─────────────────┐
│   Client App    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FastAPI Server │ ◄─── Monitoring
│   (Port 8000)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Inference Engine│
│  - Classification│
│  - Detection    │
│  - OCR          │
└─────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train a Model (if needed)

```bash
python main.py \
  --data_root ./data/cls \
  --theme classification \
  --epochs 10 \
  --n_trials 3
```

### 3. Export Model (Optional)

```bash
python production/inference/model_export.py \
  --checkpoint ./artifacts/classification/best_model.pth \
  --theme classification \
  --output_dir ./exported_models
```

### 4. Start API Server

```bash
# Set environment variables
export MODEL_PATH=./artifacts/classification/best_model.pth
export THEME=classification
export DEVICE=cuda  # or cpu

# Start server
python -m production.api.server
```

Or using uvicorn directly:

```bash
uvicorn production.api.server:app --host 0.0.0.0 --port 8000
```

### 5. Test API

```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST "http://localhost:8000/predict" \
  -F "file=@test_image.jpg"
```

## Docker Deployment

### Build Image

```bash
docker build -t convnext-api:latest .
```

### Run Container

```bash
docker run -d \
  --name convnext-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/logs:/app/logs \
  -e MODEL_PATH=/app/models/best_model.pth \
  -e THEME=classification \
  --gpus all \
  convnext-api:latest
```

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Production Components

### 1. Inference Script

Standalone inference without API:

```bash
python production/inference/inference.py \
  --checkpoint ./artifacts/classification/best_model.pth \
  --theme classification \
  --image test.jpg \
  --output predictions.json
```

Batch inference:

```bash
python production/inference/inference.py \
  --checkpoint ./artifacts/classification/best_model.pth \
  --theme classification \
  --image_dir ./test_images \
  --output batch_predictions.json
```

### 2. Input Validation

```bash
python production/validation/input_validator.py \
  --theme classification \
  --image test.jpg \
  --strict
```

### 3. Visualization

```bash
python production/utils/visualization.py \
  --predictions predictions.json \
  --images_dir ./test_images \
  --output_dir ./visualizations
```

### 4. Model Explainability (Grad-CAM)

```bash
python production/explainability/gradcam.py \
  --checkpoint ./artifacts/classification/best_model.pth \
  --image test.jpg \
  --output gradcam.png
```

### 5. Model Versioning

```bash
# Register a new model version
python production/versioning/registry.py \
  --action register \
  --model_path ./artifacts/classification/best_model.pth \
  --theme classification \
  --registry_dir ./model_registry

# List all versions
python production/versioning/registry.py \
  --action list \
  --registry_dir ./model_registry

# Promote to production
python production/versioning/registry.py \
  --action promote \
  --theme classification \
  --version v1 \
  --stage production \
  --registry_dir ./model_registry
```

### 6. A/B Testing

```bash
# Create experiment
python production/ab_testing/manager.py \
  --action create \
  --name model_comparison \
  --model_a ./models/model_v1.pth \
  --model_b ./models/model_v2.pth \
  --split 0.5

# Analyze results
python production/ab_testing/manager.py \
  --action analyze \
  --name model_comparison
```

### 7. Automated Retraining

Create `retraining_config.json`:

```json
{
  "data_root": "./data/cls",
  "theme": "classification",
  "epochs": 10,
  "n_trials": 3,
  "out_dir": "./artifacts",
  "min_accuracy_improvement": 0.02,
  "backup_previous_model": true
}
```

Run retraining:

```bash
python production/retraining/pipeline.py \
  --config retraining_config.json
```

## API Endpoints

### Health Check

```
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "theme": "classification",
  "device": "cuda",
  "timestamp": "2025-10-23T09:40:12.259Z"
}
```

### Single Prediction

```
POST /predict
```

Parameters:
- `file`: Image file (multipart/form-data)
- `img_size`: Image size for classification (default: 224)
- `det_size`: Image size for detection (default: 640)
- `ocr_h`: OCR height (default: 32)
- `ocr_w`: OCR width (default: 256)
- `conf_threshold`: Confidence threshold for detection (default: 0.5)

Response (Classification):
```json
{
  "success": true,
  "theme": "classification",
  "prediction": {
    "predicted_class": 0,
    "class_name": "class_0",
    "confidence": 0.95,
    "probabilities": {
      "class_0": 0.95,
      "class_1": 0.03,
      "class_2": 0.02
    }
  },
  "metadata": {
    "filename": "test.jpg",
    "inference_time_ms": 45.2,
    "image_size": [224, 224],
    "device": "cuda"
  },
  "timestamp": "2025-10-23T09:40:12.259Z"
}
```

### Batch Prediction

```
POST /predict/batch
```

Parameters:
- `files`: Multiple image files

### Get Metrics

```
GET /metrics
```

Response:
```json
{
  "total_predictions": 1000,
  "predictions_by_theme": {
    "classification": 1000
  },
  "avg_inference_time_ms": 45.2,
  "errors": 0,
  "start_time": "2025-10-23T09:00:00.000Z"
}
```

## Monitoring

### Prometheus Metrics

Available at `/metrics` endpoint in Prometheus format.

### Log Files

- `logs/predictions.jsonl`: All predictions
- `logs/errors.jsonl`: All errors
- `logs/metrics.json`: Aggregated metrics
- `logs/performance.jsonl`: Performance metrics

### Grafana Dashboard

Access at `http://localhost:3000` (default credentials: admin/admin)

## Best Practices

### 1. Model Management

- Always version your models
- Tag important versions (e.g., `production`, `staging`)
- Keep backups of production models
- Test thoroughly before deploying

### 2. Monitoring

- Set up alerts for:
  - High error rates
  - Slow inference times
  - Memory usage
  - Disk space

### 3. Scaling

For high-traffic scenarios:

```bash
# Run multiple workers
uvicorn production.api.server:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4
```

Or use a load balancer (nginx, HAProxy) with multiple instances.

### 4. Security

- Use HTTPS in production
- Implement authentication/authorization
- Validate and sanitize all inputs
- Rate limiting for API endpoints
- Keep dependencies updated

### 5. Performance

- Use GPU for inference when available
- Batch predictions when possible
- Cache frequently accessed models
- Monitor memory usage
- Profile slow endpoints

## Troubleshooting

### Issue: Model not loading

**Solution**: Check model path and format. Verify PyTorch version compatibility.

### Issue: Out of memory

**Solution**: Reduce batch size, use CPU inference, or upgrade hardware.

### Issue: Slow inference

**Solution**: Use GPU, export to TorchScript/ONNX, optimize model architecture.

### Issue: API not responding

**Solution**: Check logs, verify port availability, ensure model is loaded.

## Support

For issues and questions:
- Check documentation
- Review logs
- Open an issue on GitHub
- Contact the development team

## License

This project is licensed under the MIT License.
