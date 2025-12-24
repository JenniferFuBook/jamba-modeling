import time
import threading
import psutil
from typing import Dict, List, Optional


class MemoryTracker:
    """Background memory monitoring for accurate peak memory tracking."""

    def __init__(self, sampling_interval_ms: int = 100):
        """
        Initialize memory tracker.

        Args:
            sampling_interval_ms: Interval in milliseconds for memory sampling
        """
        self.sampling_interval = sampling_interval_ms / 1000.0
        self.process = psutil.Process()
        self.measurements: List[Dict[str, float]] = []
        self.monitoring = False
        self.thread: Optional[threading.Thread] = None
        self.baseline_rss = 0
        self.baseline_vms = 0

    def set_baseline(self):
        """Set the baseline memory before model loading."""
        mem_info = self.process.memory_info()
        self.baseline_rss = mem_info.rss / (1024 * 1024)
        self.baseline_vms = mem_info.vms / (1024 * 1024)

    def _monitor(self):
        """Background monitoring loop."""
        while self.monitoring:
            mem_info = self.process.memory_info()
            self.measurements.append({
                'rss_mb': mem_info.rss / (1024 * 1024),
                'vms_mb': mem_info.vms / (1024 * 1024),
                'timestamp': time.time()
            })
            time.sleep(self.sampling_interval)

    def start(self):
        """Start memory monitoring in background thread."""
        self.measurements = []
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self) -> Dict[str, float]:
        """
        Stop monitoring and return statistics.

        Returns:
            Dictionary with peak, average, and delta memory metrics
        """
        self.monitoring = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)

        if not self.measurements:
            return {
                'peak_rss_mb': 0,
                'peak_vms_mb': 0,
                'avg_rss_mb': 0,
                'avg_vms_mb': 0,
                'delta_rss_mb': 0,
                'delta_vms_mb': 0
            }

        peak_rss = max(m['rss_mb'] for m in self.measurements)
        peak_vms = max(m['vms_mb'] for m in self.measurements)
        avg_rss = sum(m['rss_mb'] for m in self.measurements) / len(self.measurements)
        avg_vms = sum(m['vms_mb'] for m in self.measurements) / len(self.measurements)

        return {
            'peak_rss_mb': round(peak_rss, 2),
            'peak_vms_mb': round(peak_vms, 2),
            'avg_rss_mb': round(avg_rss, 2),
            'avg_vms_mb': round(avg_vms, 2),
            'delta_rss_mb': round(peak_rss - self.baseline_rss, 2),
            'delta_vms_mb': round(peak_vms - self.baseline_vms, 2),
            'baseline_rss_mb': round(self.baseline_rss, 2),
            'baseline_vms_mb': round(self.baseline_vms, 2)
        }

    def get_current_memory(self) -> Dict[str, float]:
        """Get current memory usage."""
        mem_info = self.process.memory_info()
        return {
            'rss_mb': round(mem_info.rss / (1024 * 1024), 2),
            'vms_mb': round(mem_info.vms / (1024 * 1024), 2)
        }

    def reset(self):
        """Reset measurements."""
        self.measurements = []
        self.baseline_rss = 0
        self.baseline_vms = 0
