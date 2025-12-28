#!/usr/bin/env python3
"""
Test Jamba model with custom prompts.

This script loads a Jamba model and tests it on a specified prompt from a JSON file.
Works with both short and long context prompts.

Usage:
    python test_jamba.py [-p PATH] [-i ID]

Examples:
    python test_jamba.py
    python test_jamba.py -p data/short_context_prompts.json -i medium_story
    python test_jamba.py --prompts data/long_context_prompts.json --prompt-id context_8k
    python test_jamba.py -i context_1k
"""

import json
import time
import warnings
import argparse
import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Suppress expected Mamba kernel warning (we're intentionally using CPU)
warnings.filterwarnings('ignore', message='.*fast path is not available.*')
warnings.filterwarnings('ignore', message='.*Mamba kernels.*')

# Also suppress C++ level warnings by redirecting stderr during imports
def suppress_stderr():
    """Context manager to suppress stderr output."""
    class SuppressStderr:
        def __enter__(self):
            self.stderr_fd = sys.stderr.fileno()
            self.old_stderr = os.dup(self.stderr_fd)
            self.devnull = open(os.devnull, 'w')
            os.dup2(self.devnull.fileno(), self.stderr_fd)
            return self

        def __exit__(self, *args):
            os.dup2(self.old_stderr, self.stderr_fd)
            os.close(self.old_stderr)
            self.devnull.close()

    return SuppressStderr()

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_ID = "ai21labs/Jamba-tiny-dev"
USE_MAMBA_KERNELS = False

# ============================================================================
# COMMAND LINE ARGUMENTS
# ============================================================================

parser = argparse.ArgumentParser(
    description='Test Jamba model with custom context prompts',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python test_jamba.py
  python test_jamba.py -p data/short_context_prompts.json -i medium_story
  python test_jamba.py --prompts data/long_context_prompts.json --prompt-id context_8k
  python test_jamba.py -i context_1k
    """
)
parser.add_argument('-p', '--prompts', type=str, default='data/long_context_prompts.json',
                    help='Path to JSON file containing prompts (default: data/long_context_prompts.json)')
parser.add_argument('-i', '--prompt-id', type=str, default='context_8k',
                    help='ID of the prompt to use (default: context_8k)')
args = parser.parse_args()

# ============================================================================
# LOAD PROMPT
# ============================================================================

print(f"Loading prompt from {args.prompts}...")
try:
    with open(args.prompts, 'r') as f:
        prompts_data = json.load(f)
except FileNotFoundError:
    print(f"❌ Error: Prompt file not found: {args.prompts}")
    exit(1)
except json.JSONDecodeError:
    print(f"❌ Error: Invalid JSON in file: {args.prompts}")
    exit(1)

# Find the specified prompt
selected_prompt = None
for prompt in prompts_data['prompts']:
    if prompt['id'] == args.prompt_id:
        selected_prompt = prompt
        break

if not selected_prompt:
    available_prompts = [p['id'] for p in prompts_data['prompts']]
    print(f"❌ Error: Prompt '{args.prompt_id}' not found!")
    print(f"   Available prompts: {', '.join(available_prompts)}")
    exit(1)

prompt_text = selected_prompt['text']
max_tokens = selected_prompt['max_tokens']

print(f"✓ Loaded prompt: {selected_prompt['id']}")
print(f"  Estimated tokens: {selected_prompt.get('estimated_input_tokens', 'unknown')}")
print(f"  Max output tokens: {max_tokens}")

# ============================================================================
# LOAD MODEL
# ============================================================================

print(f"\nLoading model: {MODEL_ID}...")
start_load = time.perf_counter()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Suppress C++ warnings during model loading
with suppress_stderr():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map='cpu',
        use_mamba_kernels=USE_MAMBA_KERNELS,
        trust_remote_code=False
    )

load_time = time.perf_counter() - start_load
print(f"✓ Model loaded in {load_time:.2f} seconds")

# ============================================================================
# VERIFY ACTUAL TOKEN COUNT
# ============================================================================

input_token_count = len(tokenizer.encode(prompt_text))
print(f"\nInput text: {prompt_text}")
print(f"\nActual input tokens: {input_token_count:,}")

# ============================================================================
# RUN INFERENCE
# ============================================================================

print(f"\nRunning inference...")
print(f"  Generating up to {max_tokens} tokens...")

try:
    start_time = time.perf_counter()

    # Tokenize input with attention mask
    inputs = tokenizer(prompt_text, return_tensors='pt', return_attention_mask=True)

    # Generate (suppress any C++ warnings during first inference)
    with suppress_stderr():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode output
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    inference_time = time.perf_counter() - start_time

    # Calculate metrics
    total_token_count = len(tokenizer.encode(full_output))
    generated_token_count = total_token_count - input_token_count
    throughput = total_token_count / inference_time if inference_time > 0 else 0

    # ============================================================================
    # DISPLAY GENERATED CONTENT
    # ============================================================================

    # Extract just the generated portion (not the input prompt)
    prompt_end_pos = full_output.find(prompt_text) + len(prompt_text) if prompt_text in full_output else 0
    generated_content = full_output[prompt_end_pos:].strip()

    print(f"\n{'='*70}")
    print(f"GENERATED ANSWER")
    print(f"{'='*70}")
    print(generated_content)
    print(f"{'='*70}\n")
    print(f"✅ Completed in {inference_time:.1f}s ({generated_token_count} tokens)")

except Exception as e:
    print(f"\n❌ FAILED: {e}")
    print(f"\nUnexpected error - check model configuration")
    raise
