#!/usr/bin/env python3
"""
Model Versioning System
"""

import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List


class ModelVersion:
    """Represents a single model version"""
    
    def __init__(self, version: str, model_path: str, metadata: Dict[str, Any]):
        self.version = version
        self.model_path = model_path
        self.metadata = metadata
        self.created_at = metadata.get("created_at", datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "version": self.version,
            "model_path": self.model_path,
            "metadata": self.metadata,
            "created_at": self.created_at
        }


class ModelRegistry:
    """Central registry for managing model versions"""
    
    def __init__(self, registry_dir: str):
        """
        Initialize model registry
        
        Args:
            registry_dir: Directory to store registry and models
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        self.registry_file = self.registry_dir / "registry.json"
        self.models_dir = self.registry_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Load registry
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, List[Dict]]:
        """Load model registry from disk"""
        if self.registry_file.exists():
            with open(self.registry_file, "r") as f:
                return json.load(f)
        return {}
    
    def _save_registry(self):
        """Save registry to disk"""
        with open(self.registry_file, "w") as f:
            json.dump(self.registry, f, indent=2)
    
    def _compute_checksum(self, file_path: str) -> str:
        """Compute SHA256 checksum of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def register_model(self, model_path: str, theme: str, 
                      version: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> ModelVersion:
        """
        Register a new model version
        
        Args:
            model_path: Path to model checkpoint
            theme: Task type
            version: Version string (auto-generated if None)
            metadata: Additional metadata
            
        Returns:
            ModelVersion object
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Generate version if not provided
        if version is None:
            # Get existing versions for this theme
            theme_versions = self.registry.get(theme, [])
            version_num = len(theme_versions) + 1
            version = f"v{version_num}"
        
        # Compute checksum
        checksum = self._compute_checksum(model_path)
        
        # Check if this exact model already exists
        for existing in self.registry.get(theme, []):
            if existing["metadata"].get("checksum") == checksum:
                print(f"[WARNING] Model already registered as {existing['version']}")
                return ModelVersion(
                    existing["version"],
                    existing["model_path"],
                    existing["metadata"]
                )
        
        # Copy model to registry
        versioned_name = f"{theme}_{version}.pth"
        destination = self.models_dir / versioned_name
        shutil.copy(model_path, destination)
        
        # Build metadata
        full_metadata = metadata or {}
        full_metadata.update({
            "checksum": checksum,
            "original_path": str(model_path),
            "file_size_mb": model_path.stat().st_size / 1024 / 1024,
            "created_at": datetime.now().isoformat()
        })
        
        # Create version object
        model_version = ModelVersion(version, str(destination), full_metadata)
        
        # Add to registry
        if theme not in self.registry:
            self.registry[theme] = []
        
        self.registry[theme].append(model_version.to_dict())
        self._save_registry()
        
        print(f"[INFO] Registered model: {theme}/{version}")
        return model_version
    
    def get_model(self, theme: str, version: str) -> Optional[ModelVersion]:
        """Get a specific model version"""
        if theme not in self.registry:
            return None
        
        for model_data in self.registry[theme]:
            if model_data["version"] == version:
                return ModelVersion(
                    model_data["version"],
                    model_data["model_path"],
                    model_data["metadata"]
                )
        
        return None
    
    def get_latest_model(self, theme: str) -> Optional[ModelVersion]:
        """Get the latest model version for a theme"""
        if theme not in self.registry or not self.registry[theme]:
            return None
        
        # Get last registered model
        model_data = self.registry[theme][-1]
        return ModelVersion(
            model_data["version"],
            model_data["model_path"],
            model_data["metadata"]
        )
    
    def list_models(self, theme: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all models or models for a specific theme
        
        Args:
            theme: Optional theme filter
            
        Returns:
            List of model information
        """
        if theme:
            return self.registry.get(theme, [])
        
        # Return all models
        all_models = []
        for theme_name, models in self.registry.items():
            for model in models:
                model_info = model.copy()
                model_info["theme"] = theme_name
                all_models.append(model_info)
        
        return all_models
    
    def compare_versions(self, theme: str, version1: str, version2: str) -> Dict[str, Any]:
        """
        Compare two model versions
        
        Returns:
            Comparison results
        """
        model1 = self.get_model(theme, version1)
        model2 = self.get_model(theme, version2)
        
        if not model1 or not model2:
            return {"error": "One or both versions not found"}
        
        comparison = {
            "theme": theme,
            "version1": {
                "version": version1,
                "metadata": model1.metadata
            },
            "version2": {
                "version": version2,
                "metadata": model2.metadata
            },
            "differences": {}
        }
        
        # Compare metadata
        keys = set(model1.metadata.keys()) | set(model2.metadata.keys())
        for key in keys:
            val1 = model1.metadata.get(key)
            val2 = model2.metadata.get(key)
            if val1 != val2:
                comparison["differences"][key] = {
                    "version1": val1,
                    "version2": val2
                }
        
        return comparison
    
    def tag_version(self, theme: str, version: str, tags: List[str]):
        """Add tags to a model version"""
        if theme not in self.registry:
            raise ValueError(f"Theme not found: {theme}")
        
        for model in self.registry[theme]:
            if model["version"] == version:
                if "tags" not in model["metadata"]:
                    model["metadata"]["tags"] = []
                model["metadata"]["tags"].extend(tags)
                model["metadata"]["tags"] = list(set(model["metadata"]["tags"]))
                self._save_registry()
                print(f"[INFO] Tagged {theme}/{version} with: {tags}")
                return
        
        raise ValueError(f"Version not found: {version}")
    
    def promote_version(self, theme: str, version: str, stage: str):
        """
        Promote a version to a stage (e.g., 'staging', 'production')
        
        Args:
            theme: Task type
            version: Version to promote
            stage: Target stage
        """
        if theme not in self.registry:
            raise ValueError(f"Theme not found: {theme}")
        
        # Remove previous promotion from this stage
        for model in self.registry[theme]:
            if model["metadata"].get("stage") == stage:
                model["metadata"]["stage"] = None
                model["metadata"]["promoted_at"] = None
        
        # Promote new version
        for model in self.registry[theme]:
            if model["version"] == version:
                model["metadata"]["stage"] = stage
                model["metadata"]["promoted_at"] = datetime.now().isoformat()
                self._save_registry()
                print(f"[INFO] Promoted {theme}/{version} to {stage}")
                return
        
        raise ValueError(f"Version not found: {version}")
    
    def get_production_model(self, theme: str) -> Optional[ModelVersion]:
        """Get the production model for a theme"""
        if theme not in self.registry:
            return None
        
        for model_data in self.registry[theme]:
            if model_data["metadata"].get("stage") == "production":
                return ModelVersion(
                    model_data["version"],
                    model_data["model_path"],
                    model_data["metadata"]
                )
        
        return None
    
    def delete_version(self, theme: str, version: str, force: bool = False):
        """
        Delete a model version
        
        Args:
            theme: Task type
            version: Version to delete
            force: Force deletion even if in production
        """
        if theme not in self.registry:
            raise ValueError(f"Theme not found: {theme}")
        
        for i, model in enumerate(self.registry[theme]):
            if model["version"] == version:
                # Check if in production
                if model["metadata"].get("stage") == "production" and not force:
                    raise ValueError("Cannot delete production model without force=True")
                
                # Delete model file
                model_path = Path(model["model_path"])
                if model_path.exists():
                    model_path.unlink()
                
                # Remove from registry
                del self.registry[theme][i]
                self._save_registry()
                
                print(f"[INFO] Deleted {theme}/{version}")
                return
        
        raise ValueError(f"Version not found: {version}")


def main():
    """Demo model registry"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Registry Manager")
    parser.add_argument("--registry_dir", type=str, default="./model_registry")
    parser.add_argument("--action", type=str, 
                       choices=["register", "list", "get", "promote", "compare"],
                       required=True)
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--theme", type=str, help="Task theme")
    parser.add_argument("--version", type=str, help="Model version")
    parser.add_argument("--version2", type=str, help="Second version for comparison")
    parser.add_argument("--stage", type=str, help="Stage for promotion")
    
    args = parser.parse_args()
    
    registry = ModelRegistry(args.registry_dir)
    
    if args.action == "register":
        if not all([args.model_path, args.theme]):
            print("[ERROR] --model_path and --theme required")
            return
        
        model = registry.register_model(args.model_path, args.theme, args.version)
        print(f"[SUCCESS] Registered: {model.version}")
    
    elif args.action == "list":
        models = registry.list_models(args.theme)
        print(json.dumps(models, indent=2))
    
    elif args.action == "get":
        if not all([args.theme, args.version]):
            print("[ERROR] --theme and --version required")
            return
        
        model = registry.get_model(args.theme, args.version)
        if model:
            print(json.dumps(model.to_dict(), indent=2))
        else:
            print("[ERROR] Model not found")
    
    elif args.action == "promote":
        if not all([args.theme, args.version, args.stage]):
            print("[ERROR] --theme, --version, and --stage required")
            return
        
        registry.promote_version(args.theme, args.version, args.stage)
    
    elif args.action == "compare":
        if not all([args.theme, args.version, args.version2]):
            print("[ERROR] --theme, --version, and --version2 required")
            return
        
        comparison = registry.compare_versions(args.theme, args.version, args.version2)
        print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
