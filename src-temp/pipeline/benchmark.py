"""GPU power monitoring and benchmarking for pipeline."""

import time
import threading
from contextlib import contextmanager
from typing import Callable, Dict, Any, Optional
import numpy as np

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    try:
        # Try nvidia-ml-py (newer package name)
        import pynvml
        PYNVML_AVAILABLE = True
    except ImportError:
        PYNVML_AVAILABLE = False
        print("Warning: pynvml/nvidia-ml-py not available. Power monitoring will be disabled.")


class GPUPowerMonitor:
    """Context manager for monitoring GPU power consumption."""
    
    def __init__(self, gpu_id: int = 0, sample_rate: float = 0.1):
        """
        Initialize GPU power monitor.
        
        Args:
            gpu_id: GPU device ID
            sample_rate: Sampling rate in seconds (default 100ms = 0.1s)
        """
        self.gpu_id = gpu_id
        self.sample_rate = sample_rate
        self.power_samples: list[float] = []
        self.timestamps: list[float] = []
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            except Exception as e:
                print(f"Warning: Failed to initialize NVML: {e}")
                self.handle = None
        else:
            self.handle = None
    
    def _monitor_loop(self):
        """Internal monitoring loop."""
        while self.is_monitoring:
            if self.handle is not None:
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # mW to W
                    timestamp = time.time()
                    self.power_samples.append(power)
                    self.timestamps.append(timestamp)
                except Exception:
                    timestamp = time.time()
                    self.power_samples.append(0.0)
                    self.timestamps.append(timestamp)
            else:
                timestamp = time.time()
                self.power_samples.append(0.0)
                self.timestamps.append(timestamp)
            
            time.sleep(self.sample_rate)
    
    def __enter__(self):
        """Start monitoring."""
        self.start_time = time.time()
        self.is_monitoring = True
        self.power_samples = []
        self.timestamps = []
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop monitoring."""
        self.is_monitoring = False
        if self.monitor_thread is not None:
            self.monitor_thread.join(timeout=1.0)
        self.end_time = time.time()
    
    @property
    def total_energy_j(self) -> float:
        """Calculate total energy consumption in Joules."""
        if len(self.power_samples) < 2:
            return 0.0
        
        power_array = np.array(self.power_samples)
        time_array = np.array(self.timestamps)
        
        if len(time_array) > 1:
            dt = np.diff(time_array)
            avg_power = (power_array[:-1] + power_array[1:]) / 2.0
            energy = np.sum(avg_power * dt)
            return float(energy)
        
        return 0.0
    
    @property
    def avg_power_w(self) -> float:
        """Calculate average power consumption in Watts."""
        if len(self.power_samples) == 0:
            return 0.0
        return float(np.mean(self.power_samples))
    
    @property
    def duration_s(self) -> float:
        """Get monitoring duration in seconds."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time


class PipelineBenchmark:
    """Benchmarking infrastructure for pipeline."""
    
    def __init__(self, gpu_id: int = 0, n_warmup: int = 5):
        """
        Initialize benchmarker.
        
        Args:
            gpu_id: GPU device ID
            n_warmup: Number of warmup runs for JIT compilation
        """
        self.gpu_id = gpu_id
        self.n_warmup = n_warmup
    
    def run(
        self,
        denoise_fn: Callable[[np.ndarray], np.ndarray],
        noisy_image: np.ndarray,
        n_runs: int = 1,
    ) -> Dict[str, Any]:
        """
        Run denoising function with power monitoring.
        
        Args:
            denoise_fn: Denoising function
            noisy_image: Input noisy image
            n_runs: Number of benchmark runs
        
        Returns:
            Dictionary with output, power metrics, and performance metrics
        """
        # Warmup runs
        for _ in range(self.n_warmup):
            _ = denoise_fn(noisy_image)
        
        # Benchmark runs
        times = []
        energies = []
        powers = []
        denoised = None
        
        for i in range(n_runs):
            monitor = GPUPowerMonitor(gpu_id=self.gpu_id)
            
            with monitor:
                start = time.time()
                denoised = denoise_fn(noisy_image)
                end = time.time()
            
            times.append((end - start) * 1000.0)  # ms
            energies.append(monitor.total_energy_j)
            powers.append(monitor.avg_power_w)
        
        # Calculate statistics
        return {
            'output': denoised,
            'power': {
                'mean_energy_j': float(np.mean(energies)),
                'std_energy_j': float(np.std(energies)),
                'mean_power_w': float(np.mean(powers)),
                'std_power_w': float(np.std(powers)),
            },
            'performance': {
                'mean_time_ms': float(np.mean(times)),
                'std_time_ms': float(np.std(times)),
                'n_runs': n_runs,
            },
        }

