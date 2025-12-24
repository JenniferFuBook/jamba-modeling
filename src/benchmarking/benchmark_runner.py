import gc
import logging
from typing import Dict, List, Any
from .memory_tracker import MemoryTracker
from .metrics import Metrics
from ..models.base_model import BaseModel


logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Orchestrates benchmark execution for language models."""

    def __init__(self, model: BaseModel, config: Dict[str, Any]):
        """
        Initialize benchmark runner.

        Args:
            model: Language model to benchmark
            config: Benchmark configuration dictionary
        """
        self.model = model
        self.config = config
        self.memory_tracker = MemoryTracker(
            sampling_interval_ms=config.get('memory', {}).get('sampling_interval_ms', 100)
        )
        self.warmup_iterations = config.get('warmup_iterations', 3)
        self.benchmark_iterations = config.get('benchmark_iterations', 20)
        self.generation_params = config.get('generation', {})

    def run_warmup(self, prompt: str, max_tokens: int):
        """
        Run warmup iterations to stabilize performance.

        Args:
            prompt: Test prompt
            max_tokens: Maximum tokens to generate
        """
        logger.info(f"Running {self.warmup_iterations} warmup iterations...")

        for i in range(self.warmup_iterations):
            try:
                self.model.generate(prompt, max_tokens, **self.generation_params)
                logger.debug(f"Warmup iteration {i+1}/{self.warmup_iterations} completed")
            except Exception as e:
                logger.warning(f"Warmup iteration {i+1} failed: {e}")

        gc.collect()

    def benchmark_prompt(self, prompt: str, max_tokens: int) -> Dict[str, Any]:
        """
        Benchmark a single prompt with multiple iterations.

        Args:
            prompt: Test prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Dictionary with benchmark results
        """
        logger.info(f"Benchmarking prompt (max_tokens={max_tokens})...")

        self.run_warmup(prompt, max_tokens)

        results = []

        for i in range(self.benchmark_iterations):
            try:
                self.memory_tracker.start()

                generated_text, inference_time = self.model.generate(
                    prompt,
                    max_tokens,
                    **self.generation_params
                )

                memory_stats = self.memory_tracker.stop()

                input_tokens = Metrics.count_tokens(self.model.tokenizer, prompt)
                total_tokens = Metrics.count_tokens(self.model.tokenizer, generated_text)
                generated_tokens = total_tokens - input_tokens

                throughput = Metrics.calculate_throughput(total_tokens, inference_time)

                results.append({
                    'inference_time': inference_time,
                    'total_tokens': total_tokens,
                    'generated_tokens': generated_tokens,
                    'throughput': throughput,
                    'memory': memory_stats
                })

                logger.debug(f"Iteration {i+1}/{self.benchmark_iterations}: {inference_time:.4f}s, {throughput:.2f} tok/s")

            except Exception as e:
                logger.error(f"Benchmark iteration {i+1} failed: {e}")
                continue

        aggregated = Metrics.aggregate_results(results)

        aggregated['prompt'] = prompt
        aggregated['max_tokens'] = max_tokens
        aggregated['iterations'] = len(results)
        aggregated['individual_results'] = results

        return aggregated

    def run_full_benchmark(self, prompts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run full benchmark suite across all prompts.

        Args:
            prompts: List of prompt dictionaries

        Returns:
            Complete benchmark results
        """
        logger.info(f"Starting full benchmark with {len(prompts)} prompts...")

        self.memory_tracker.set_baseline()

        load_time = self.model.load_model()
        logger.info(f"Model loaded in {load_time:.2f} seconds")

        load_memory = self.memory_tracker.get_current_memory()

        model_size = self.model.get_model_size()

        prompt_results = []

        for idx, prompt_config in enumerate(prompts):
            prompt_text = prompt_config['text']
            max_tokens = prompt_config['max_tokens']
            prompt_id = prompt_config.get('id', f'prompt_{idx}')
            category = prompt_config.get('category', 'unknown')

            logger.info(f"Benchmarking prompt '{prompt_id}' (category: {category})...")

            result = self.benchmark_prompt(prompt_text, max_tokens)
            result['prompt_id'] = prompt_id
            result['category'] = category

            prompt_results.append(result)

            gc.collect()

        all_latencies = []
        all_throughputs = []
        all_memory_peaks = []

        for pr in prompt_results:
            if 'latency' in pr:
                all_latencies.extend([pr['latency']['mean']])
            if 'throughput' in pr:
                all_throughputs.extend([pr['throughput']['mean']])
            if 'individual_results' in pr:
                for ir in pr['individual_results']:
                    if 'memory' in ir and 'peak_rss_mb' in ir['memory']:
                        all_memory_peaks.append(ir['memory']['peak_rss_mb'])

        overall_stats = {
            'latency': Metrics.calculate_latency_stats(all_latencies) if all_latencies else {},
            'memory': {
                'baseline_mb': load_memory['rss_mb'],
                'peak_rss_mb': round(max(all_memory_peaks), 2) if all_memory_peaks else 0,
                'avg_peak_rss_mb': round(sum(all_memory_peaks) / len(all_memory_peaks), 2) if all_memory_peaks else 0
            }
        }

        if all_throughputs:
            overall_stats['throughput'] = {
                'mean': round(sum(all_throughputs) / len(all_throughputs), 2),
                'min': round(min(all_throughputs), 2),
                'max': round(max(all_throughputs), 2)
            }

        return {
            'model_id': self.model.model_id,
            'load_time': round(load_time, 4),
            'model_size': model_size,
            'prompt_results': prompt_results,
            'overall_stats': overall_stats
        }
