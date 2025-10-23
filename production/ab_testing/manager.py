#!/usr/bin/env python3
"""
A/B Testing Framework for Model Comparison
"""

import json
import random
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from collections import defaultdict
import threading


class ABTestManager:
    """Manage A/B testing experiments for models"""
    
    def __init__(self, config_path: str):
        """
        Initialize A/B test manager
        
        Args:
            config_path: Path to A/B test configuration
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.experiments = {}
        self.results = defaultdict(lambda: {"A": [], "B": []})
        self.lock = threading.Lock()
        
        # Load experiments
        self._load_experiments()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load A/B test configuration"""
        if not self.config_path.exists():
            default_config = {
                "experiments": [],
                "default_split": 0.5,
                "results_dir": "./ab_test_results",
                "min_samples": 100
            }
            
            with open(self.config_path, "w") as f:
                json.dump(default_config, f, indent=2)
            
            return default_config
        
        with open(self.config_path, "r") as f:
            return json.load(f)
    
    def _load_experiments(self):
        """Load active experiments"""
        for exp in self.config.get("experiments", []):
            self.experiments[exp["name"]] = exp
    
    def create_experiment(self, name: str, model_a_path: str, model_b_path: str,
                         split: float = 0.5, metadata: Optional[Dict] = None):
        """
        Create a new A/B test experiment
        
        Args:
            name: Experiment name
            model_a_path: Path to model A checkpoint
            model_b_path: Path to model B checkpoint
            split: Traffic split (0.0 to 1.0, default 0.5)
            metadata: Additional metadata
        """
        experiment = {
            "name": name,
            "model_a": model_a_path,
            "model_b": model_b_path,
            "split": split,
            "created_at": datetime.now().isoformat(),
            "active": True,
            "metadata": metadata or {}
        }
        
        self.experiments[name] = experiment
        
        # Save to config
        self.config["experiments"].append(experiment)
        with open(self.config_path, "w") as f:
            json.dump(self.config, f, indent=2)
        
        print(f"[INFO] Created A/B test experiment: {name}")
        return experiment
    
    def assign_variant(self, experiment_name: str, user_id: Optional[str] = None) -> str:
        """
        Assign a user to a variant (A or B)
        
        Args:
            experiment_name: Name of experiment
            user_id: Optional user identifier for consistent assignment
            
        Returns:
            Variant name ("A" or "B")
        """
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment not found: {experiment_name}")
        
        experiment = self.experiments[experiment_name]
        split = experiment.get("split", 0.5)
        
        if user_id:
            # Consistent hashing for same user
            hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
            return "A" if (hash_value % 100) / 100 < split else "B"
        else:
            # Random assignment
            return "A" if random.random() < split else "B"
    
    def get_model_path(self, experiment_name: str, variant: str) -> str:
        """Get model path for a variant"""
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment not found: {experiment_name}")
        
        experiment = self.experiments[experiment_name]
        
        if variant == "A":
            return experiment["model_a"]
        elif variant == "B":
            return experiment["model_b"]
        else:
            raise ValueError(f"Invalid variant: {variant}")
    
    def record_result(self, experiment_name: str, variant: str, 
                     metrics: Dict[str, float]):
        """
        Record result for a variant
        
        Args:
            experiment_name: Name of experiment
            variant: Variant ("A" or "B")
            metrics: Performance metrics
        """
        with self.lock:
            result = {
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics
            }
            
            self.results[experiment_name][variant].append(result)
            
            # Save results periodically
            if len(self.results[experiment_name][variant]) % 10 == 0:
                self._save_results(experiment_name)
    
    def _save_results(self, experiment_name: str):
        """Save experiment results"""
        results_dir = Path(self.config.get("results_dir", "./ab_test_results"))
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / f"{experiment_name}_results.json"
        
        with open(results_file, "w") as f:
            json.dump(dict(self.results[experiment_name]), f, indent=2)
    
    def analyze_experiment(self, experiment_name: str) -> Dict[str, Any]:
        """
        Analyze experiment results
        
        Returns:
            Analysis summary
        """
        if experiment_name not in self.results:
            return {"error": "No results available"}
        
        results = self.results[experiment_name]
        
        analysis = {
            "experiment": experiment_name,
            "variant_a": self._analyze_variant(results["A"]),
            "variant_b": self._analyze_variant(results["B"]),
            "comparison": {}
        }
        
        # Compare variants
        min_samples = self.config.get("min_samples", 100)
        if len(results["A"]) >= min_samples and len(results["B"]) >= min_samples:
            analysis["comparison"] = self._compare_variants(
                results["A"], results["B"]
            )
            analysis["recommendation"] = self._get_recommendation(analysis["comparison"])
        else:
            analysis["comparison"]["status"] = "insufficient_data"
            analysis["recommendation"] = "Continue collecting data"
        
        return analysis
    
    def _analyze_variant(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze results for a single variant"""
        if not results:
            return {"samples": 0}
        
        # Extract all metrics
        all_metrics = defaultdict(list)
        for result in results:
            for key, value in result["metrics"].items():
                all_metrics[key].append(value)
        
        # Calculate statistics
        stats = {
            "samples": len(results),
            "metrics": {}
        }
        
        for metric, values in all_metrics.items():
            stats["metrics"][metric] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "std": self._calculate_std(values)
            }
        
        return stats
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def _compare_variants(self, results_a: List[Dict], 
                         results_b: List[Dict]) -> Dict[str, Any]:
        """Compare two variants statistically"""
        comparison = {}
        
        # Get common metrics
        metrics_a = defaultdict(list)
        metrics_b = defaultdict(list)
        
        for result in results_a:
            for key, value in result["metrics"].items():
                metrics_a[key].append(value)
        
        for result in results_b:
            for key, value in result["metrics"].items():
                metrics_b[key].append(value)
        
        # Compare each metric
        for metric in set(metrics_a.keys()) & set(metrics_b.keys()):
            mean_a = sum(metrics_a[metric]) / len(metrics_a[metric])
            mean_b = sum(metrics_b[metric]) / len(metrics_b[metric])
            
            improvement = ((mean_b - mean_a) / mean_a * 100) if mean_a != 0 else 0
            
            comparison[metric] = {
                "variant_a_mean": mean_a,
                "variant_b_mean": mean_b,
                "improvement_pct": improvement,
                "winner": "B" if mean_b > mean_a else "A"
            }
        
        return comparison
    
    def _get_recommendation(self, comparison: Dict[str, Any]) -> str:
        """Get recommendation based on comparison"""
        if not comparison:
            return "Insufficient data for recommendation"
        
        # Count wins for each variant
        wins_a = 0
        wins_b = 0
        
        for metric, data in comparison.items():
            if data["winner"] == "A":
                wins_a += 1
            else:
                wins_b += 1
        
        if wins_b > wins_a:
            return "Deploy Variant B (statistically better)"
        elif wins_a > wins_b:
            return "Keep Variant A (statistically better)"
        else:
            return "No clear winner, continue testing"
    
    def stop_experiment(self, experiment_name: str):
        """Stop an experiment"""
        if experiment_name in self.experiments:
            self.experiments[experiment_name]["active"] = False
            self._save_results(experiment_name)
            
            # Generate final report
            analysis = self.analyze_experiment(experiment_name)
            
            report_path = Path(self.config.get("results_dir", "./ab_test_results"))
            report_path = report_path / f"{experiment_name}_final_report.json"
            
            with open(report_path, "w") as f:
                json.dump(analysis, f, indent=2)
            
            print(f"[INFO] Stopped experiment: {experiment_name}")
            print(f"[INFO] Final report: {report_path}")
            
            return analysis


def main():
    """Demo A/B testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="A/B Testing Manager")
    parser.add_argument("--config", type=str, default="./ab_test_config.json")
    parser.add_argument("--action", type=str, choices=["create", "analyze", "stop"], required=True)
    parser.add_argument("--name", type=str, help="Experiment name")
    parser.add_argument("--model_a", type=str, help="Path to model A")
    parser.add_argument("--model_b", type=str, help="Path to model B")
    parser.add_argument("--split", type=float, default=0.5, help="Traffic split")
    
    args = parser.parse_args()
    
    manager = ABTestManager(args.config)
    
    if args.action == "create":
        if not all([args.name, args.model_a, args.model_b]):
            print("[ERROR] --name, --model_a, and --model_b required for create")
            return
        
        manager.create_experiment(args.name, args.model_a, args.model_b, args.split)
    
    elif args.action == "analyze":
        if not args.name:
            print("[ERROR] --name required for analyze")
            return
        
        analysis = manager.analyze_experiment(args.name)
        print(json.dumps(analysis, indent=2))
    
    elif args.action == "stop":
        if not args.name:
            print("[ERROR] --name required for stop")
            return
        
        manager.stop_experiment(args.name)


if __name__ == "__main__":
    main()
