FROM python:3.11-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

FROM base AS models

ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    print('Downloading GPT-2...'); \
    AutoModelForCausalLM.from_pretrained('gpt2'); \
    AutoTokenizer.from_pretrained('gpt2'); \
    print('GPT-2 downloaded successfully')"

RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; print('Downloading Jamba-tiny-dev...'); AutoModelForCausalLM.from_pretrained('ai21labs/Jamba-tiny-dev', use_mamba_kernels=False, trust_remote_code=False); AutoTokenizer.from_pretrained('ai21labs/Jamba-tiny-dev'); print('Jamba-tiny-dev downloaded successfully')" || echo "Warning: Jamba download failed, will retry at runtime"

FROM models AS final

WORKDIR /app

COPY config/ ./config/
COPY data/ ./data/
COPY src/ ./src/

RUN mkdir -p results

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

VOLUME ["/app/results"]

CMD ["python", "-m", "src.main"]
