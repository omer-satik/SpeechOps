from prometheus_client import Counter, Histogram, Gauge
import time

# Request counter
request_count = Counter(
    'speechops_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

# Request duration
request_duration = Histogram(
    'speechops_request_duration_seconds',
    'API request duration',
    ['method', 'endpoint'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0)
)

# Model prediction counter
predictions_total = Counter(
    'speechops_predictions_total',
    'Total predictions',
    ['status']
)

# Model prediction duration
prediction_duration = Histogram(
    'speechops_prediction_duration_seconds',
    'Model prediction duration',
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0)
)

# Audio file size
audio_file_size = Histogram(
    'speechops_audio_size_bytes',
    'Processed audio file size',
    buckets=(1000, 10000, 100000, 1000000, 10000000)
)

# System health
api_health = Gauge(
    'speechops_api_health',
    'API health status (1=healthy, 0=unhealthy)'
)

# Is model loaded
model_loaded = Gauge(
    'speechops_model_loaded',
    'Is model loaded in memory (1=yes, 0=no)'
)

# Processing success rate
processing_success_rate = Gauge(
    'speechops_processing_success_rate',
    'Processing success rate (0-1)'
)

# Real Time Factor (Processing Time / Input Duration)
processing_rtf = Histogram(
    'speechops_processing_rtf',
    'Real Time Factor (Processing Time / Input Duration)',
    buckets=(0.1, 0.5, 0.8, 1.0, 1.2, 2.0, 5.0)
)

