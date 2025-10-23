# API Reference

## Overview

The ConvNeXt Multi-Task API provides RESTful endpoints for classification, object detection, and OCR inference.

Base URL: `http://localhost:8000`

## Authentication

Currently, the API does not require authentication. For production, implement:
- API keys
- OAuth 2.0
- JWT tokens

## Rate Limiting

Default: 100 requests per minute per IP

## Endpoints

### GET /

Root endpoint with API information.

**Response:**
```json
{
  "message": "ConvNeXt Multi-Task API",
  "version": "1.0.0",
  "theme": "classification",
  "endpoints": {
    "health": "/health",
    "predict": "/predict",
    "predict_batch": "/predict/batch",
    "metrics": "/metrics"
  }
}
```

---

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "theme": "classification",
  "device": "cuda",
  "timestamp": "2025-10-23T09:40:12.259Z"
}
```

**Status Codes:**
- `200 OK`: Service is healthy
- `503 Service Unavailable`: Service is unhealthy

---

### POST /predict

Perform inference on a single image.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| file | File | Yes | - | Image file (JPEG, PNG, etc.) |
| img_size | int | No | 224 | Image size for classification |
| det_size | int | No | 640 | Image size for detection |
| ocr_h | int | No | 32 | OCR image height |
| ocr_w | int | No | 256 | OCR image width |
| conf_threshold | float | No | 0.5 | Confidence threshold for detection |

**Request (curl):**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@image.jpg" \
  -F "img_size=224"
```

**Request (Python):**
```python
import requests

with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f},
        data={'img_size': 224}
    )

result = response.json()
```

**Response (Classification):**
```json
{
  "success": true,
  "theme": "classification",
  "prediction": {
    "predicted_class": 0,
    "class_name": "cat",
    "confidence": 0.956,
    "probabilities": {
      "cat": 0.956,
      "dog": 0.032,
      "bird": 0.012
    }
  },
  "metadata": {
    "filename": "image.jpg",
    "inference_time_ms": 45.2,
    "image_size": [224, 224],
    "device": "cuda"
  },
  "timestamp": "2025-10-23T09:40:12.259Z"
}
```

**Response (Object Detection):**
```json
{
  "success": true,
  "theme": "object_detection",
  "prediction": {
    "detections": [
      {
        "bbox": [10, 20, 150, 200],
        "label": 1,
        "class_name": "person",
        "confidence": 0.92
      },
      {
        "bbox": [200, 50, 350, 300],
        "label": 2,
        "class_name": "car",
        "confidence": 0.88
      }
    ],
    "num_objects": 2
  },
  "metadata": {
    "filename": "image.jpg",
    "inference_time_ms": 120.5
  },
  "timestamp": "2025-10-23T09:40:12.259Z"
}
```

**Response (OCR):**
```json
{
  "success": true,
  "theme": "ocr",
  "prediction": {
    "text": "HELLO WORLD",
    "confidence": 0.94
  },
  "metadata": {
    "filename": "image.jpg",
    "inference_time_ms": 35.8
  },
  "timestamp": "2025-10-23T09:40:12.259Z"
}
```

**Status Codes:**
- `200 OK`: Successful prediction
- `400 Bad Request`: Invalid input
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Model not loaded

---

### POST /predict/batch

Perform inference on multiple images.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| files | File[] | Yes | Multiple image files |
| img_size | int | No | Image size for classification |
| det_size | int | No | Image size for detection |

**Request (curl):**
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg"
```

**Request (Python):**
```python
import requests

files = [
    ('files', open('image1.jpg', 'rb')),
    ('files', open('image2.jpg', 'rb')),
    ('files', open('image3.jpg', 'rb'))
]

response = requests.post(
    'http://localhost:8000/predict/batch',
    files=files
)

results = response.json()
```

**Response:**
```json
{
  "total": 3,
  "successful": 3,
  "failed": 0,
  "results": [
    {
      "success": true,
      "theme": "classification",
      "prediction": { ... },
      "metadata": { ... }
    },
    {
      "success": true,
      "theme": "classification",
      "prediction": { ... },
      "metadata": { ... }
    },
    {
      "success": true,
      "theme": "classification",
      "prediction": { ... },
      "metadata": { ... }
    }
  ]
}
```

---

### GET /metrics

Get API usage metrics.

**Response:**
```json
{
  "total_predictions": 1000,
  "predictions_by_theme": {
    "classification": 800,
    "object": 150,
    "ocr": 50
  },
  "avg_inference_time_ms": 45.2,
  "errors": 5,
  "start_time": "2025-10-23T09:00:00.000Z"
}
```

---

### GET /model/info

Get information about the loaded model.

**Response:**
```json
{
  "theme": "classification",
  "model_path": "/app/models/best_model.pth",
  "device": "cuda",
  "metadata": {
    "variant": "convnext_tiny",
    "num_classes": 10
  }
}
```

---

## Error Responses

All endpoints return errors in this format:

```json
{
  "error": "Error message",
  "detail": "Detailed error information",
  "timestamp": "2025-10-23T09:40:12.259Z"
}
```

## Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid input |
| 404 | Not Found - Endpoint does not exist |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Model not loaded |

## Usage Examples

### Python Client

```python
import requests
from pathlib import Path

class ConvNeXtClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def health(self):
        """Check API health"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def predict(self, image_path, **kwargs):
        """Predict on single image"""
        with open(image_path, 'rb') as f:
            response = requests.post(
                f"{self.base_url}/predict",
                files={'file': f},
                data=kwargs
            )
        return response.json()
    
    def predict_batch(self, image_paths, **kwargs):
        """Predict on multiple images"""
        files = [('files', open(p, 'rb')) for p in image_paths]
        try:
            response = requests.post(
                f"{self.base_url}/predict/batch",
                files=files,
                data=kwargs
            )
            return response.json()
        finally:
            for _, f in files:
                f.close()

# Usage
client = ConvNeXtClient()

# Health check
print(client.health())

# Single prediction
result = client.predict('test.jpg')
print(result)

# Batch prediction
results = client.predict_batch(['img1.jpg', 'img2.jpg', 'img3.jpg'])
print(results)
```

### JavaScript/Node.js Client

```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

class ConvNeXtClient {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
  }

  async health() {
    const response = await axios.get(`${this.baseUrl}/health`);
    return response.data;
  }

  async predict(imagePath, options = {}) {
    const formData = new FormData();
    formData.append('file', fs.createReadStream(imagePath));
    
    Object.entries(options).forEach(([key, value]) => {
      formData.append(key, value);
    });

    const response = await axios.post(
      `${this.baseUrl}/predict`,
      formData,
      { headers: formData.getHeaders() }
    );
    
    return response.data;
  }
}

// Usage
const client = new ConvNeXtClient();

// Health check
client.health().then(console.log);

// Prediction
client.predict('test.jpg', { img_size: 224 })
  .then(result => console.log(result))
  .catch(error => console.error(error));
```

## Best Practices

1. **Batch Processing**: Use `/predict/batch` for multiple images to reduce overhead
2. **Error Handling**: Always handle errors gracefully
3. **Timeouts**: Set appropriate timeouts for your requests
4. **Retry Logic**: Implement exponential backoff for retries
5. **Image Size**: Optimize image sizes before sending to reduce bandwidth

## Support

For API issues or questions:
- Check the documentation
- Review error messages
- Contact support team
