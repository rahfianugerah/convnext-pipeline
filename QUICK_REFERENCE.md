# Quick Reference Guide

## 🚀 Common Commands

### Training
```bash
# Classification
python main.py --data_root ./data/cls --theme classification --epochs 10 --n_trials 3

# Object Detection
python main.py --data_root ./data/det --theme object --epochs 15 --n_trials 3

# OCR
python main.py --data_root ./data/ocr --theme ocr --epochs 15 --n_trials 5
```

### Inference
```bash
# Single image
python production/inference/inference.py \
  --checkpoint ./artifacts/classification/best_model.pth \
  --theme classification \
  --image test.jpg

# Batch processing
python production/inference/inference.py \
  --checkpoint ./artifacts/classification/best_model.pth \
  --theme classification \
  --image_dir ./test_images \
  --output results.json
```

### API Server
```bash
# Start server
export MODEL_PATH=./artifacts/classification/best_model.pth
export THEME=classification
python -m production.api.server

# Test endpoints
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -F "file=@test.jpg"
```

### Docker
```bash
# Build
docker build -t convnext-api:latest .

# Run
docker run -d -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -e MODEL_PATH=/app/models/best_model.pth \
  -e THEME=classification \
  convnext-api:latest

# Compose
docker-compose up -d
docker-compose logs -f
docker-compose down
```

### Model Export
```bash
# Export to all formats
python production/inference/model_export.py \
  --checkpoint ./artifacts/classification/best_model.pth \
  --theme classification \
  --format all

# ONNX only
python production/inference/model_export.py \
  --checkpoint ./artifacts/classification/best_model.pth \
  --theme classification \
  --format onnx
```

### Validation
```bash
# Validate single image
python production/validation/input_validator.py \
  --theme classification \
  --image test.jpg \
  --strict

# Validate multiple
python production/validation/input_validator.py \
  --theme classification \
  --images img1.jpg img2.jpg img3.jpg
```

### Visualization
```bash
# Visualize predictions
python production/utils/visualization.py \
  --predictions predictions.json \
  --images_dir ./test_images \
  --output_dir ./visualizations \
  --max_images 10
```

### Explainability
```bash
# Generate Grad-CAM
python production/explainability/gradcam.py \
  --checkpoint ./artifacts/classification/best_model.pth \
  --image test.jpg \
  --output gradcam.png
```

### Model Versioning
```bash
# Register version
python production/versioning/registry.py \
  --action register \
  --model_path ./artifacts/classification/best_model.pth \
  --theme classification \
  --version v1

# List versions
python production/versioning/registry.py \
  --action list \
  --theme classification

# Promote to production
python production/versioning/registry.py \
  --action promote \
  --theme classification \
  --version v1 \
  --stage production

# Get specific version
python production/versioning/registry.py \
  --action get \
  --theme classification \
  --version v1

# Compare versions
python production/versioning/registry.py \
  --action compare \
  --theme classification \
  --version v1 \
  --version2 v2
```

### A/B Testing
```bash
# Create experiment
python production/ab_testing/manager.py \
  --action create \
  --name model_comparison \
  --model_a ./models/v1.pth \
  --model_b ./models/v2.pth \
  --split 0.5

# Analyze results
python production/ab_testing/manager.py \
  --action analyze \
  --name model_comparison

# Stop experiment
python production/ab_testing/manager.py \
  --action stop \
  --name model_comparison
```

### Automated Retraining
```bash
# Create config
cat > retraining_config.json << EOF
{
  "data_root": "./data/cls",
  "theme": "classification",
  "epochs": 10,
  "n_trials": 3,
  "out_dir": "./artifacts",
  "min_accuracy_improvement": 0.02
}
EOF

# Run retraining
python production/retraining/pipeline.py \
  --config retraining_config.json
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=production --cov-report=html

# Specific test
pytest tests/test_production.py::test_input_validator -v

# Skip slow tests
pytest tests/ -v -m "not slow"
```

## 📊 API Quick Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch prediction |
| GET | `/metrics` | API metrics |
| GET | `/model/info` | Model info |

### Example Requests

**Python:**
```python
import requests

# Single prediction
with open('test.jpg', 'rb') as f:
    r = requests.post('http://localhost:8000/predict', files={'file': f})
    print(r.json())

# Batch prediction
files = [('files', open(f'img{i}.jpg', 'rb')) for i in range(3)]
r = requests.post('http://localhost:8000/predict/batch', files=files)
print(r.json())
```

**cURL:**
```bash
# Health check
curl http://localhost:8000/health

# Predict
curl -X POST http://localhost:8000/predict \
  -F "file=@test.jpg" \
  -F "img_size=224"

# Batch predict
curl -X POST http://localhost:8000/predict/batch \
  -F "files=@img1.jpg" \
  -F "files=@img2.jpg"
```

## 🔧 Configuration Files

### Docker Compose
```yaml
# docker-compose.yml
services:
  api:
    build: .
    ports: ["8000:8000"]
    volumes:
      - ./models:/app/models:ro
    environment:
      - MODEL_PATH=/app/models/best_model.pth
      - THEME=classification
```

### Retraining Config
```json
{
  "data_root": "./data/cls",
  "theme": "classification",
  "epochs": 10,
  "n_trials": 3,
  "min_accuracy_improvement": 0.02,
  "backup_previous_model": true
}
```

### A/B Test Config
```json
{
  "experiments": [],
  "default_split": 0.5,
  "results_dir": "./ab_test_results",
  "min_samples": 100
}
```

## 🐛 Troubleshooting

**Model not loading:**
```bash
# Check model path
ls -lh ./artifacts/classification/best_model.pth

# Verify checkpoint
python -c "import torch; print(torch.load('best_model.pth').keys())"
```

**Out of memory:**
```bash
# Use CPU
export DEVICE=cpu
python -m production.api.server

# Reduce batch size
# Edit inference.py, set batch_size=1
```

**Port already in use:**
```bash
# Use different port
python -m production.api.server --port 8001

# Or kill existing process
lsof -ti:8000 | xargs kill -9
```

## 📈 Monitoring

**View logs:**
```bash
# Predictions
tail -f logs/predictions.jsonl

# Errors
tail -f logs/errors.jsonl

# Metrics
cat logs/metrics.json | jq
```

**Prometheus metrics:**
```bash
curl http://localhost:8000/metrics
```

**Grafana:**
- URL: http://localhost:3000
- Default credentials: admin/admin

## 🔒 Security Notes

For production deployment:
1. Enable HTTPS
2. Add authentication (API keys, OAuth)
3. Implement rate limiting
4. Validate and sanitize all inputs
5. Keep dependencies updated
6. Use secrets management
7. Enable CORS properly
8. Set up firewall rules

## 📚 Documentation Links

- [Deployment Guide](docs/DEPLOYMENT.md)
- [API Reference](docs/API.md)
- [Production README](docs/PRODUCTION_README.md)
- [Implementation Summary](IMPLEMENTATION_SUMMARY.md)

## 🆘 Support

**Issues:**
- Check documentation
- Review logs
- Search existing issues
- Open new issue with details

**Community:**
- GitHub Discussions
- Stack Overflow tag
- Discord/Slack channel
