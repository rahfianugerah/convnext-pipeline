#!/usr/bin/env python3
"""
Prediction Logging and Monitoring System
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from collections import defaultdict
import threading


class PredictionLogger:
    """Logs predictions for monitoring and analysis"""
    
    def __init__(self, log_dir: str):
        """
        Initialize logger
        
        Args:
            log_dir: Directory to store logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log files
        self.prediction_log = self.log_dir / "predictions.jsonl"
        self.metrics_log = self.log_dir / "metrics.json"
        
        # In-memory metrics
        self.metrics = {
            "total_predictions": 0,
            "predictions_by_theme": defaultdict(int),
            "avg_inference_time_ms": [],
            "errors": 0,
            "start_time": datetime.now().isoformat()
        }
        
        # Thread lock for safe concurrent access
        self.lock = threading.Lock()
    
    def log_prediction(self, theme: str, prediction: Dict[str, Any], 
                      metadata: Dict[str, Any] = None):
        """
        Log a prediction
        
        Args:
            theme: Task type
            prediction: Prediction results
            metadata: Additional metadata
        """
        with self.lock:
            # Create log entry
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "theme": theme,
                "prediction": prediction,
                "metadata": metadata or {}
            }
            
            # Write to log file (append mode)
            with open(self.prediction_log, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            
            # Update metrics
            self.metrics["total_predictions"] += 1
            self.metrics["predictions_by_theme"][theme] += 1
            
            if metadata and "inference_time_ms" in metadata:
                self.metrics["avg_inference_time_ms"].append(metadata["inference_time_ms"])
            
            # Periodically save metrics
            if self.metrics["total_predictions"] % 10 == 0:
                self._save_metrics()
    
    def log_error(self, theme: str, error: str, metadata: Dict[str, Any] = None):
        """Log an error"""
        with self.lock:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "theme": theme,
                "error": error,
                "metadata": metadata or {}
            }
            
            error_log = self.log_dir / "errors.jsonl"
            with open(error_log, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            
            self.metrics["errors"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        with self.lock:
            metrics = dict(self.metrics)
            
            # Calculate average inference time
            if self.metrics["avg_inference_time_ms"]:
                avg_time = sum(self.metrics["avg_inference_time_ms"]) / len(self.metrics["avg_inference_time_ms"])
                metrics["avg_inference_time_ms"] = round(avg_time, 2)
            else:
                metrics["avg_inference_time_ms"] = 0
            
            # Convert defaultdict to regular dict
            metrics["predictions_by_theme"] = dict(metrics["predictions_by_theme"])
            
            return metrics
    
    def _save_metrics(self):
        """Save metrics to file"""
        metrics = self.get_metrics()
        with open(self.metrics_log, "w") as f:
            json.dump(metrics, f, indent=2)
    
    def close(self):
        """Cleanup and save final metrics"""
        self._save_metrics()


class PerformanceMonitor:
    """Monitor model performance metrics"""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.performance_log = self.log_dir / "performance.jsonl"
        self.lock = threading.Lock()
    
    def log_performance(self, theme: str, metrics: Dict[str, float]):
        """
        Log performance metrics
        
        Args:
            theme: Task type
            metrics: Performance metrics (accuracy, loss, etc.)
        """
        with self.lock:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "theme": theme,
                "metrics": metrics
            }
            
            with open(self.performance_log, "a") as f:
                f.write(json.dumps(entry) + "\n")
    
    def get_recent_performance(self, theme: str = None, limit: int = 100) -> List[Dict]:
        """Get recent performance logs"""
        if not self.performance_log.exists():
            return []
        
        entries = []
        with open(self.performance_log, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                if theme is None or entry["theme"] == theme:
                    entries.append(entry)
        
        return entries[-limit:]


class PrometheusExporter:
    """Export metrics in Prometheus format"""
    
    def __init__(self, logger: PredictionLogger):
        self.logger = logger
    
    def export_metrics(self) -> str:
        """Export metrics in Prometheus text format"""
        metrics = self.logger.get_metrics()
        
        lines = []
        
        # Total predictions
        lines.append("# HELP predictions_total Total number of predictions")
        lines.append("# TYPE predictions_total counter")
        lines.append(f"predictions_total {metrics['total_predictions']}")
        
        # Predictions by theme
        lines.append("# HELP predictions_by_theme_total Predictions by theme")
        lines.append("# TYPE predictions_by_theme_total counter")
        for theme, count in metrics.get("predictions_by_theme", {}).items():
            lines.append(f'predictions_by_theme_total{{theme="{theme}"}} {count}')
        
        # Average inference time
        lines.append("# HELP inference_time_ms Average inference time in milliseconds")
        lines.append("# TYPE inference_time_ms gauge")
        lines.append(f"inference_time_ms {metrics.get('avg_inference_time_ms', 0)}")
        
        # Errors
        lines.append("# HELP errors_total Total number of errors")
        lines.append("# TYPE errors_total counter")
        lines.append(f"errors_total {metrics.get('errors', 0)}")
        
        return "\n".join(lines)


def main():
    """Demo logger usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test prediction logger")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Log directory")
    args = parser.parse_args()
    
    # Initialize logger
    logger = PredictionLogger(args.log_dir)
    
    # Log some test predictions
    for i in range(5):
        logger.log_prediction(
            theme="classification",
            prediction={"class": i, "confidence": 0.95},
            metadata={"inference_time_ms": 50 + i * 10}
        )
    
    # Get metrics
    metrics = logger.get_metrics()
    print("[Metrics]")
    print(json.dumps(metrics, indent=2))
    
    # Close logger
    logger.close()
    
    print(f"\n[INFO] Logs saved to {args.log_dir}")


if __name__ == "__main__":
    main()
