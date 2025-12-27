import time
import torch
import gc
import warnings
import sys
import os
from typing import Tuple, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base_model import BaseModel

# Suppress Mamba kernel warnings globally (expected on CPU-only systems)
warnings.filterwarnings('ignore', message='.*fast path is not available.*')
warnings.filterwarnings('ignore', message='.*Mamba kernels.*')


class JambaModel(BaseModel):
    """Jamba model wrapper for benchmarking."""

    def load_model(self) -> float:
        """Load Jamba model and tokenizer."""
        start_time = time.perf_counter()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            padding_side='left'
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        use_mamba_kernels = self.model_config.get('use_mamba_kernels', False)
        trust_remote_code = self.model_config.get('trust_remote_code', False)

        # Suppress Mamba kernel warning (expected on CPU)
        # This warning is printed by the C++ extension, so we need to redirect stderr
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*fast path is not available.*')

            # Temporarily redirect stderr to suppress C++ warnings
            stderr_fd = sys.stderr.fileno()
            with open(os.devnull, 'w') as devnull:
                old_stderr = os.dup(stderr_fd)
                os.dup2(devnull.fileno(), stderr_fd)
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        dtype=torch.float32,
                        low_cpu_mem_usage=True,
                        use_mamba_kernels=use_mamba_kernels,
                        trust_remote_code=trust_remote_code
                    )
                finally:
                    # Restore stderr
                    os.dup2(old_stderr, stderr_fd)
                    os.close(old_stderr)

        self.model.to(self.device)
        self.model.eval()

        end_time = time.perf_counter()
        return end_time - start_time

    def generate(self, prompt: str, max_tokens: int, **generation_kwargs) -> Tuple[str, float]:
        """Generate text using Jamba."""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, return_attention_mask=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        start_time = time.perf_counter()

        # Suppress Mamba kernel warnings during generation
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*fast path is not available.*')

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **generation_kwargs
                )

        end_time = time.perf_counter()

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        inference_time = end_time - start_time

        return generated_text, inference_time

    def get_model_size(self) -> Dict[str, Any]:
        """Get Jamba model size information."""
        if self.model is None:
            return {"error": "Model not loaded"}

        num_params = sum(p.numel() for p in self.model.parameters())
        num_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        param_size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)

        return {
            "total_parameters": num_params,
            "trainable_parameters": num_trainable_params,
            "model_size_mb": round(param_size_mb, 2),
            "dtype": str(next(self.model.parameters()).dtype)
        }

    def cleanup(self):
        """Clean up Jamba model resources."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
