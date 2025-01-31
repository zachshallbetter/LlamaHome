"""System monitoring implementation."""

import asyncio
import logging
from datetime import datetime

import psutil
import torch
from prometheus_client import Counter, Gauge, Histogram
from pydantic import BaseModel


class ResourceMetrics(BaseModel):
    """Resource metrics data."""

    cpu_usage: float
    memory_usage: float
    gpu_usage: float | None
    gpu_memory: float | None
    disk_usage: float
    network_io: dict[str, float]


class SystemMetrics(BaseModel):
    """System metrics data."""

    timestamp: datetime
    resources: ResourceMetrics
    error_count: int
    request_count: int
    average_response_time: float


class AlertConfig(BaseModel):
    """Alert configuration."""

    cpu_threshold: float = 90.0
    memory_threshold: float = 90.0
    gpu_threshold: float = 90.0
    error_rate_threshold: float = 5.0
    response_time_threshold: float = 1.0


class SystemMonitor:
    """System monitoring implementation."""

    def __init__(self, alert_config: AlertConfig, metrics_interval: int = 60) -> None:
        """Initialize system monitor.

        Args:
            alert_config: Alert configuration
            metrics_interval: Metrics collection interval in seconds
        """
        self.config = alert_config
        self.metrics_interval = metrics_interval
        self.logger = logging.getLogger("system_monitor")

        # Prometheus metrics
        self.cpu_gauge = Gauge("cpu_usage", "CPU usage percentage")
        self.memory_gauge = Gauge("memory_usage", "Memory usage percentage")
        self.gpu_gauge = Gauge("gpu_usage", "GPU usage percentage")
        self.error_counter = Counter("error_count", "Number of errors")
        self.request_histogram = Histogram(
            "request_duration_seconds", "Request duration in seconds"
        )

    async def start_monitoring(self) -> None:
        """Start the monitoring loop."""
        while True:
            try:
                # Collect metrics
                metrics = await self.collect_metrics()

                # Update Prometheus metrics
                self.update_prometheus_metrics(metrics)

                # Check thresholds and alert if necessary
                await self.check_alerts(metrics)

                # Log metrics
                self.logger.info(f"System metrics: {metrics.json()}")

                # Wait for next collection
                await asyncio.sleep(self.metrics_interval)

            except Exception as e:
                self.logger.error(f"Monitoring error: {str(e)}")
                await asyncio.sleep(self.metrics_interval)

    async def collect_metrics(self) -> SystemMetrics:
        """Collect system metrics."""
        try:
            # Collect CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)

            # Collect memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Collect GPU metrics if available
            gpu_metrics = self._collect_gpu_metrics()

            # Collect disk metrics
            disk = psutil.disk_usage("/")
            disk_percent = disk.percent

            # Collect network I/O
            network = psutil.net_io_counters()
            network_metrics = {
                "bytes_sent": float(network.bytes_sent),
                "bytes_recv": float(network.bytes_recv),
            }

            # Create resource metrics
            resources = ResourceMetrics(
                cpu_usage=cpu_percent,
                memory_usage=memory_percent,
                gpu_usage=gpu_metrics.get("usage"),
                gpu_memory=gpu_metrics.get("memory"),
                disk_usage=disk_percent,
                network_io=network_metrics,
            )

            # Create system metrics
            return SystemMetrics(
                timestamp=datetime.utcnow(),
                resources=resources,
                error_count=int(self.error_counter._value.get()),
                request_count=int(self.request_histogram._count.get()),
                average_response_time=self._calculate_average_response_time(),
            )

        except Exception as e:
            self.logger.error(f"Error collecting metrics: {str(e)}")
            raise

    def _collect_gpu_metrics(self) -> dict[str, float]:
        """Collect GPU metrics if available."""
        try:
            if torch.cuda.is_available():
                gpu_metrics = {}
                for i in range(torch.cuda.device_count()):
                    gpu_metrics[f"gpu_{i}"] = {
                        "usage": float(torch.cuda.utilization(i)),
                        "memory": float(
                            torch.cuda.memory_allocated(i)
                            / torch.cuda.max_memory_allocated(i)
                            * 100
                        ),
                    }
                return gpu_metrics
            return {}
        except Exception:
            return {}

    def _calculate_average_response_time(self) -> float:
        """Calculate average response time from histogram."""
        try:
            if self.request_histogram._count.get() > 0:
                return float(
                    self.request_histogram._sum.get()
                    / self.request_histogram._count.get()
                )
            return 0.0
        except Exception:
            return 0.0

    def update_prometheus_metrics(self, metrics: SystemMetrics) -> None:
        """Update Prometheus metrics."""
        self.cpu_gauge.set(metrics.resources.cpu_usage)
        self.memory_gauge.set(metrics.resources.memory_usage)
        if metrics.resources.gpu_usage is not None:
            self.gpu_gauge.set(metrics.resources.gpu_usage)

    async def check_alerts(self, metrics: SystemMetrics) -> None:
        """Check metrics against thresholds and generate alerts."""
        alerts = []

        # Check CPU usage
        if metrics.resources.cpu_usage > self.config.cpu_threshold:
            alerts.append(f"High CPU usage: {metrics.resources.cpu_usage}%")

        # Check memory usage
        if metrics.resources.memory_usage > self.config.memory_threshold:
            alerts.append(f"High memory usage: {metrics.resources.memory_usage}%")

        # Check GPU usage
        if (
            metrics.resources.gpu_usage
            and metrics.resources.gpu_usage > self.config.gpu_threshold
        ):
            alerts.append(f"High GPU usage: {metrics.resources.gpu_usage}%")

        # Check error rate
        error_rate = self._calculate_error_rate(metrics)
        if error_rate > self.config.error_rate_threshold:
            alerts.append(f"High error rate: {error_rate}%")

        # Check response time
        if metrics.average_response_time > self.config.response_time_threshold:
            alerts.append(
                f"High average response time: {metrics.average_response_time}s"
            )

        # Log alerts
        if alerts:
            alert_message = "System Alerts:\n" + "\n".join(alerts)
            self.logger.warning(alert_message)

    def _calculate_error_rate(self, metrics: SystemMetrics) -> float:
        """Calculate error rate as percentage of requests."""
        if metrics.request_count > 0:
            return (metrics.error_count / metrics.request_count) * 100
        return 0.0

    async def record_request(self, duration: float, error: bool = False) -> None:
        """Record a request for metrics."""
        self.request_histogram.observe(duration)
        if error:
            self.error_counter.inc()
