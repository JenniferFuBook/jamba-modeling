from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any


class BaseModel(ABC):
    """Abstract base class for language models."""

    def __init__(self, model_id: str, device: str = "cpu", **kwargs):
        """
        Initialize the base model.

        Args:
            model_id: Hugging Face model identifier
            device: Device to run inference on (cpu/cuda)
            **kwargs: Additional model-specific arguments
        """
        self.model_id = model_id
        self.device = device
        self.model = None
        self.tokenizer = None
        self.model_config = kwargs

    @abstractmethod
    def load_model(self) -> float:
        """
        Load the model and tokenizer.

        Returns:
            Loading time in seconds
        """
        pass

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int, **generation_kwargs) -> Tuple[str, float]:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate
            **generation_kwargs: Additional generation parameters (temperature, top_p, etc.)

        Returns:
            Tuple of (generated_text, inference_time_seconds)
        """
        pass

    @abstractmethod
    def get_model_size(self) -> Dict[str, Any]:
        """
        Get model size information.

        Returns:
            Dictionary with model size metrics (parameters, file size, etc.)
        """
        pass

    @abstractmethod
    def cleanup(self):
        """
        Clean up model resources and free memory.
        """
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(model_id={self.model_id}, device={self.device})"
