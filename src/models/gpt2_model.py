import time
import torch
import gc
from typing import Tuple, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base_model import BaseModel


class GPT2Model(BaseModel):
    """GPT-2 model wrapper for benchmarking."""

    def load_model(self) -> float:
        """Load GPT-2 model and tokenizer."""
        start_time = time.perf_counter()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            padding_side='left'
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.float32,
            low_cpu_mem_usage=True
        )

        self.model.to(self.device)
        self.model.eval()

        end_time = time.perf_counter()
        return end_time - start_time

    def generate(self, prompt: str, max_tokens: int, **generation_kwargs) -> Tuple[str, float]:
        """Generate text using GPT-2."""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, return_attention_mask=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        start_time = time.perf_counter()

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
        """Get GPT-2 model size information."""
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
        """Clean up GPT-2 model resources."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
