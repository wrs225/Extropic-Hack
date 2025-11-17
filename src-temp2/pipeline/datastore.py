"""Image data storage and management."""

from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
from PIL import Image
import json
from datetime import datetime


class ImageDataStore:
    """
    Manages image data storage for the pipeline.
    
    Handles:
    - Loading images from files
    - Saving processed images
    - Organizing results by timestamp/experiment
    - Metadata storage
    """
    
    def __init__(self, base_dir: Path = None):
        """
        Initialize data store.
        
        Args:
            base_dir: Base directory for data storage. If None, uses 'data' in current directory.
        """
        if base_dir is None:
            # Default to 'data' directory relative to pipeline location
            base_dir = Path(__file__).parent.parent / "data"
        
        self.base_dir = Path(base_dir)
        self.input_dir = self.base_dir / "input"   # Input images go here
        self.output_dir = self.base_dir / "output"  # Output results go here
        self.raw_dir = self.base_dir / "raw"        # Raw/original images
        
        # Create directories
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
    
    def load_image(
        self,
        path: Path,
        target_size: Optional[Tuple[int, int]] = (256, 256),
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Load an image from file.
        
        Args:
            path: Path to image file
            target_size: Target size (H, W), None to keep original
            normalize: If True, normalize to [0, 1]
        
        Returns:
            Image array as float32
        """
        img = Image.open(path)
        
        # Convert to grayscale if needed
        if img.mode != 'L':
            img = img.convert('L')
        
        # Resize if needed
        if target_size is not None:
            img = img.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
        
        # Convert to numpy
        img_array = np.array(img, dtype=np.float32)
        
        # Normalize
        if normalize:
            img_array = img_array / 255.0
        
        return img_array
    
    def save_image(
        self,
        image: np.ndarray,
        path: Path,
        denormalize: bool = True,
    ) -> None:
        """
        Save an image to file.
        
        Args:
            image: Image array, shape (H, W)
            path: Output file path
            denormalize: If True, assume [0, 1] and convert to [0, 255]
        """
        # Ensure 2D
        if len(image.shape) > 2:
            image = image.squeeze()
        
        # Denormalize if needed
        if denormalize:
            image = np.clip(image, 0.0, 1.0)
            image = (image * 255.0).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Save
        img = Image.fromarray(image, mode='L')
        img.save(path)
    
    def save_results(
        self,
        results: dict,
        experiment_name: Optional[str] = None,
    ) -> Path:
        """
        Save pipeline results to organized directory.
        
        Args:
            results: Results dictionary from pipeline
            experiment_name: Optional experiment name
        
        Returns:
            Path to saved results directory
        """
        # Create experiment directory in output folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name:
            exp_dir = self.output_dir / f"{experiment_name}_{timestamp}"
        else:
            exp_dir = self.output_dir / f"experiment_{timestamp}"
        
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save images
        self.save_image(results['clean'], exp_dir / "clean.png")
        self.save_image(results['noisy'], exp_dir / "noisy.png")
        self.save_image(results['denoised'], exp_dir / "denoised.png")
        
        # Save metrics as JSON
        metrics_path = exp_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(results['metrics'], f, indent=2)
        
        # Save full results
        results_path = exp_dir / "results.json"
        # Convert numpy arrays to lists for JSON
        json_results = {
            'metrics': results['metrics'],
            'power': results['power'],
            'performance': results['performance'],
            'metadata': results['metadata'],
            'timestamp': timestamp,
        }
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        return exp_dir
    
    def list_experiments(self) -> List[Path]:
        """List all experiment directories."""
        return sorted(self.output_dir.glob("experiment_*"), reverse=True)

