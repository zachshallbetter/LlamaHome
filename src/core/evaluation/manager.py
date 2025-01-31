"""Evaluation manager implementation."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config.base import BaseConfig
from ..utils import LogManager, LogTemplates

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)


class EvaluationConfig(BaseConfig):
    """Evaluation configuration."""

    metrics: List[str] = ["accuracy", "loss", "perplexity"]
    batch_size: int = 32
    num_samples: Optional[int] = None
    save_results: bool = True
    output_dir: Path = Path("evaluation_results")
    log_progress: bool = True
    device: str = "cuda"


class EvaluationManager:
    """Manages model evaluation process."""

    def __init__(self, config: EvaluationConfig) -> None:
        """Initialize evaluation manager.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.results: Dict[str, Any] = {}
        self._setup_output_dir()

    def _setup_output_dir(self) -> None:
        """Set up output directory."""
        if self.config.save_results:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)

    async def evaluate(
        self, model: Any, dataset: Any, metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluate model on dataset.
        
        Args:
            model: Model to evaluate
            dataset: Evaluation dataset
            metrics: Optional list of metrics to compute
            
        Returns:
            Dictionary of evaluation results
        """
        metrics = metrics or self.config.metrics
        self.results = {}

        try:
            # Compute each requested metric
            for metric in metrics:
                if hasattr(self, f"_compute_{metric}"):
                    metric_fn = getattr(self, f"_compute_{metric}")
                    self.results[metric] = await metric_fn(model, dataset)
                else:
                    logger.warning(f"Metric not implemented: {metric}")

            # Save results if configured
            if self.config.save_results:
                self._save_results()

            return self.results

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise EvaluationError(f"Evaluation failed: {e}") from e

    async def _compute_accuracy(self, model: Any, dataset: Any) -> float:
        """Compute accuracy metric.
        
        Args:
            model: Model to evaluate
            dataset: Evaluation dataset
            
        Returns:
            Accuracy score
        """
        correct = 0
        total = 0

        for batch in dataset:
            predictions = model(batch["input_ids"])
            labels = batch["labels"]
            correct += (predictions.argmax(-1) == labels).sum().item()
            total += labels.numel()

        return correct / total if total > 0 else 0.0

    async def _compute_loss(self, model: Any, dataset: Any) -> float:
        """Compute loss metric.
        
        Args:
            model: Model to evaluate
            dataset: Evaluation dataset
            
        Returns:
            Loss value
        """
        total_loss = 0.0
        num_batches = 0

        for batch in dataset:
            loss = model(batch["input_ids"], labels=batch["labels"]).loss
            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else float("inf")

    async def _compute_perplexity(self, model: Any, dataset: Any) -> float:
        """Compute perplexity metric.
        
        Args:
            model: Model to evaluate
            dataset: Evaluation dataset
            
        Returns:
            Perplexity score
        """
        import torch
        import math

        total_loss = 0.0
        total_tokens = 0

        for batch in dataset:
            with torch.no_grad():
                outputs = model(batch["input_ids"], labels=batch["labels"])
                total_loss += outputs.loss.item() * batch["labels"].ne(-100).sum().item()
                total_tokens += batch["labels"].ne(-100).sum().item()

        return math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")

    def _save_results(self) -> None:
        """Save evaluation results."""
        import json

        results_file = self.config.output_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)

    def get_results(self) -> Dict[str, Any]:
        """Get evaluation results.
        
        Returns:
            Dictionary of evaluation results
        """
        return self.results.copy()


class EvaluationError(Exception):
    """Evaluation related errors."""

    pass 