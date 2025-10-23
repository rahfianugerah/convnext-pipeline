#!/usr/bin/env python3
"""
Automated Model Retraining Pipeline
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import subprocess
import sys


class RetrainingPipeline:
    """Automated pipeline for model retraining"""
    
    def __init__(self, config_path: str):
        """
        Initialize retraining pipeline
        
        Args:
            config_path: Path to retraining configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Create directories
        self.logs_dir = Path(self.config.get("logs_dir", "./retraining_logs"))
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.models_dir = Path(self.config.get("models_dir", "./retrained_models"))
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load retraining configuration"""
        if not self.config_path.exists():
            # Create default config
            default_config = {
                "data_root": "./data/cls",
                "theme": "classification",
                "epochs": 10,
                "n_trials": 3,
                "out_dir": "./retrained_models",
                "logs_dir": "./retraining_logs",
                "models_dir": "./retrained_models",
                "min_accuracy_improvement": 0.02,
                "backup_previous_model": True,
                "notification_email": None
            }
            
            with open(self.config_path, "w") as f:
                json.dump(default_config, f, indent=2)
            
            print(f"[INFO] Created default config at {self.config_path}")
            return default_config
        
        with open(self.config_path, "r") as f:
            return json.load(f)
    
    def check_trigger_conditions(self) -> bool:
        """
        Check if retraining should be triggered
        
        Returns:
            True if retraining should start
        """
        # Check various conditions
        conditions = []
        
        # 1. Check if performance has degraded
        if "performance_threshold" in self.config:
            # Would check recent performance metrics
            conditions.append(True)
        
        # 2. Check if new data is available
        if "data_root" in self.config:
            data_root = Path(self.config["data_root"])
            if data_root.exists():
                conditions.append(True)
        
        # 3. Check if scheduled retraining is due
        if "retrain_schedule" in self.config:
            # Would check if it's time based on schedule
            conditions.append(True)
        
        # For now, return True if any condition is met
        return any(conditions)
    
    def backup_current_model(self):
        """Backup current best model"""
        if not self.config.get("backup_previous_model", True):
            return
        
        theme = self.config.get("theme", "classification")
        current_model = Path(self.config.get("out_dir", "./artifacts")) / theme / "best_model.pth"
        
        if current_model.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.models_dir / f"backup_{theme}_{timestamp}.pth"
            shutil.copy(current_model, backup_path)
            print(f"[INFO] Backed up current model to {backup_path}")
            return backup_path
        
        return None
    
    def run_training(self) -> Dict[str, Any]:
        """
        Run training process
        
        Returns:
            Training results
        """
        print("[INFO] Starting retraining...")
        
        # Build training command
        cmd = [
            sys.executable, "main.py",
            "--data_root", self.config["data_root"],
            "--theme", self.config["theme"],
            "--epochs", str(self.config.get("epochs", 10)),
            "--n_trials", str(self.config.get("n_trials", 3)),
            "--out_dir", self.config["out_dir"]
        ]
        
        # Add optional parameters
        if "img_size" in self.config:
            cmd.extend(["--img_size", str(self.config["img_size"])])
        if "det_size" in self.config:
            cmd.extend(["--det_size", str(self.config["det_size"])])
        if "workers" in self.config:
            cmd.extend(["--workers", str(self.config["workers"])])
        
        # Create log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"retraining_{timestamp}.log"
        
        # Run training
        with open(log_file, "w") as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )
        
        print(f"[INFO] Training log saved to {log_file}")
        
        return {
            "success": result.returncode == 0,
            "log_file": str(log_file),
            "timestamp": timestamp,
            "return_code": result.returncode
        }
    
    def evaluate_new_model(self) -> Dict[str, Any]:
        """
        Evaluate newly trained model
        
        Returns:
            Evaluation metrics
        """
        theme = self.config.get("theme", "classification")
        test_metrics_path = Path(self.config.get("out_dir", "./artifacts")) / theme / "test_metrics.json"
        
        if test_metrics_path.exists():
            with open(test_metrics_path, "r") as f:
                metrics = json.load(f)
            return metrics
        
        return {}
    
    def compare_models(self, old_metrics: Dict[str, Any], 
                      new_metrics: Dict[str, Any]) -> bool:
        """
        Compare old and new model performance
        
        Returns:
            True if new model is better
        """
        theme = self.config.get("theme")
        min_improvement = self.config.get("min_accuracy_improvement", 0.02)
        
        if theme == "classification":
            old_acc = old_metrics.get("acc", 0)
            new_acc = new_metrics.get("acc", 0)
            improvement = new_acc - old_acc
            
            print(f"[INFO] Old accuracy: {old_acc:.4f}")
            print(f"[INFO] New accuracy: {new_acc:.4f}")
            print(f"[INFO] Improvement: {improvement:.4f}")
            
            return improvement >= min_improvement
            
        elif theme == "object":
            old_map = old_metrics.get("mAP@0.5:0.95", 0)
            new_map = new_metrics.get("mAP@0.5:0.95", 0)
            improvement = new_map - old_map
            
            print(f"[INFO] Old mAP: {old_map:.4f}")
            print(f"[INFO] New mAP: {new_map:.4f}")
            print(f"[INFO] Improvement: {improvement:.4f}")
            
            return improvement >= min_improvement
            
        elif theme == "ocr":
            old_cer = old_metrics.get("CER", 1.0)
            new_cer = new_metrics.get("CER", 1.0)
            improvement = old_cer - new_cer  # Lower is better
            
            print(f"[INFO] Old CER: {old_cer:.4f}")
            print(f"[INFO] New CER: {new_cer:.4f}")
            print(f"[INFO] Improvement: {improvement:.4f}")
            
            return improvement >= min_improvement
        
        return False
    
    def deploy_new_model(self):
        """Deploy newly trained model"""
        print("[INFO] Deploying new model...")
        
        # In a real production system, this would:
        # 1. Update model registry
        # 2. Notify relevant services
        # 3. Perform rolling update
        # 4. Monitor deployment health
        
        deployment_info = {
            "timestamp": datetime.now().isoformat(),
            "theme": self.config.get("theme"),
            "status": "deployed"
        }
        
        deployment_log = self.logs_dir / "deployment_history.jsonl"
        with open(deployment_log, "a") as f:
            f.write(json.dumps(deployment_info) + "\n")
        
        print("[INFO] New model deployed successfully")
    
    def rollback(self, backup_path: Path):
        """Rollback to previous model"""
        print("[WARNING] Rolling back to previous model...")
        
        theme = self.config.get("theme", "classification")
        current_model = Path(self.config.get("out_dir", "./artifacts")) / theme / "best_model.pth"
        
        if backup_path and backup_path.exists():
            shutil.copy(backup_path, current_model)
            print(f"[INFO] Rolled back to {backup_path}")
        else:
            print("[ERROR] No backup available for rollback")
    
    def run_pipeline(self):
        """Run complete retraining pipeline"""
        print("\n" + "="*60)
        print("AUTOMATED RETRAINING PIPELINE")
        print("="*60 + "\n")
        
        # Check if retraining should be triggered
        if not self.check_trigger_conditions():
            print("[INFO] No retraining needed at this time")
            return
        
        print("[INFO] Retraining conditions met, starting pipeline...\n")
        
        # Load old metrics
        theme = self.config.get("theme", "classification")
        old_metrics_path = Path(self.config.get("out_dir", "./artifacts")) / theme / "test_metrics.json"
        old_metrics = {}
        if old_metrics_path.exists():
            with open(old_metrics_path, "r") as f:
                old_metrics = json.load(f)
        
        # Backup current model
        backup_path = self.backup_current_model()
        
        # Run training
        training_result = self.run_training()
        
        if not training_result["success"]:
            print("[ERROR] Training failed, see log for details")
            return
        
        print("[INFO] Training completed successfully\n")
        
        # Evaluate new model
        new_metrics = self.evaluate_new_model()
        
        # Compare models
        if old_metrics:
            is_better = self.compare_models(old_metrics, new_metrics)
            
            if is_better:
                print("\n[SUCCESS] New model is better, deploying...")
                self.deploy_new_model()
            else:
                print("\n[WARNING] New model is not better, rolling back...")
                if backup_path:
                    self.rollback(backup_path)
        else:
            print("\n[INFO] No previous metrics found, deploying new model...")
            self.deploy_new_model()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED")
        print("="*60 + "\n")


def main():
    """Run retraining pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated Retraining Pipeline")
    parser.add_argument("--config", type=str, default="./retraining_config.json",
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    pipeline = RetrainingPipeline(args.config)
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
