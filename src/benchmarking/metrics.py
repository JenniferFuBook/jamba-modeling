import numpy as np
from typing import List, Dict, Any


class Metrics:
    """Metrics collection and calculation utilities."""

    @staticmethod
    def calculate_latency_stats(latencies: List[float]) -> Dict[str, float]:
        """
        Calculate statistical measures for latency.

        Args:
            latencies: List of latency measurements in seconds

        Returns:
            Dictionary with mean, median, std, min, max
        """
        if not latencies:
            return {
                'mean': 0,
                'median': 0,
                'std': 0,
                'min': 0,
                'max': 0,
                'count': 0
            }

        return {
            'mean': round(np.mean(latencies), 4),
            'median': round(np.median(latencies), 4),
            'std': round(np.std(latencies), 4),
            'min': round(np.min(latencies), 4),
            'max': round(np.max(latencies), 4),
            'count': len(latencies)
        }

    @staticmethod
    def calculate_throughput(total_tokens: int, total_time: float) -> float:
        """
        Calculate throughput in tokens per second.

        Args:
            total_tokens: Total number of tokens processed
            total_time: Total time in seconds

        Returns:
            Tokens per second
        """
        if total_time == 0:
            return 0
        return round(total_tokens / total_time, 2)

    @staticmethod
    def count_tokens(tokenizer, text: str) -> int:
        """
        Count tokens in text using tokenizer.

        Args:
            tokenizer: Hugging Face tokenizer
            text: Text to tokenize

        Returns:
            Number of tokens
        """
        return len(tokenizer.encode(text))

    @staticmethod
    def aggregate_results(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate results from multiple benchmark runs.

        Args:
            results_list: List of individual result dictionaries

        Returns:
            Aggregated statistics
        """
        if not results_list:
            return {}

        latencies = [r['inference_time'] for r in results_list if 'inference_time' in r]

        throughputs = []
        total_tokens_list = []
        generated_tokens_list = []

        for r in results_list:
            if 'throughput' in r:
                throughputs.append(r['throughput'])
            if 'total_tokens' in r:
                total_tokens_list.append(r['total_tokens'])
            if 'generated_tokens' in r:
                generated_tokens_list.append(r['generated_tokens'])

        aggregated = {
            'latency': Metrics.calculate_latency_stats(latencies),
        }

        if throughputs:
            aggregated['throughput'] = {
                'mean': round(np.mean(throughputs), 2),
                'median': round(np.median(throughputs), 2),
                'std': round(np.std(throughputs), 2),
                'min': round(np.min(throughputs), 2),
                'max': round(np.max(throughputs), 2)
            }

        if total_tokens_list:
            aggregated['avg_total_tokens'] = round(np.mean(total_tokens_list), 2)

        if generated_tokens_list:
            aggregated['avg_generated_tokens'] = round(np.mean(generated_tokens_list), 2)

        return aggregated

    @staticmethod
    def compare_models(model1_results: Dict[str, Any], model2_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare results between two models.

        Args:
            model1_results: Results from first model
            model2_results: Results from second model

        Returns:
            Comparison statistics and winner analysis
        """
        comparison = {}

        if 'latency' in model1_results and 'latency' in model2_results:
            latency1 = model1_results['latency']['mean']
            latency2 = model2_results['latency']['mean']

            comparison['latency_winner'] = 'model1' if latency1 < latency2 else 'model2'
            comparison['latency_ratio'] = round(max(latency1, latency2) / min(latency1, latency2), 2) if min(latency1, latency2) > 0 else 0
            comparison['latency_difference_ms'] = round(abs(latency1 - latency2) * 1000, 2)

        if 'throughput' in model1_results and 'throughput' in model2_results:
            throughput1 = model1_results['throughput']['mean']
            throughput2 = model2_results['throughput']['mean']

            comparison['throughput_winner'] = 'model1' if throughput1 > throughput2 else 'model2'
            comparison['throughput_ratio'] = round(max(throughput1, throughput2) / min(throughput1, throughput2), 2) if min(throughput1, throughput2) > 0 else 0

        if 'memory' in model1_results and 'memory' in model2_results:
            mem1 = model1_results['memory'].get('peak_rss_mb', 0)
            mem2 = model2_results['memory'].get('peak_rss_mb', 0)

            comparison['memory_winner'] = 'model1' if mem1 < mem2 else 'model2'
            comparison['memory_ratio'] = round(max(mem1, mem2) / min(mem1, mem2), 2) if min(mem1, mem2) > 0 else 0
            comparison['memory_difference_mb'] = round(abs(mem1 - mem2), 2)

        if 'load_time' in model1_results and 'load_time' in model2_results:
            load1 = model1_results['load_time']
            load2 = model2_results['load_time']

            comparison['load_time_winner'] = 'model1' if load1 < load2 else 'model2'
            comparison['load_time_ratio'] = round(max(load1, load2) / min(load1, load2), 2) if min(load1, load2) > 0 else 0

        return comparison
