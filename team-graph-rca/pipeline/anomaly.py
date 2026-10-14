"""
Anomaly detection utilities for cloud-native metrics.
"""

CPU_THRESHOLD = 85.0
ERROR_RATE_THRESHOLD = 0.05

def detect_anomaly(cpu_usage: float, error_rate: float):
    """
    Evaluates metric values against thresholds and returns (is_anomalous, anomaly_type).
    """
    if cpu_usage > CPU_THRESHOLD:
        return True, "cpu_exhaustion"
    if error_rate > ERROR_RATE_THRESHOLD:
        return True, "high_error_rate"
    return False, None
