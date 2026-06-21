# =============================================================================
# Dockerfile — RAG-приложение RAG-Chatbot
# =============================================================================
# Base: nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
# Python: 3.10 (чистый, без conda)
# PyTorch: 2.4.0+cu121 (через pip, явный индекс)
# Совместимо с RTX 3090 и vLLM на CUDA 12.1
# =============================================================================

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Python 3.10 + pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Делаем python3.10 дефолтным
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Обновляем pip
RUN pip install --no-cache-dir --upgrade pip

# Устанавливаем torch отдельным шагом — кешируется независимо от остальных зависимостей
RUN pip install --no-cache-dir \
    torch==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Остальные зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/models

CMD ["python", "main.py"]
