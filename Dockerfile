FROM python:3.11-slim

# cu128 = CUDA 12.8 build, needed for RTX 50-series GPUs (PyTorch >= 2.7).
# Teammates without an NVIDIA GPU can build with:  --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    NLTK_DATA=/usr/local/share/nltk_data

# graphviz = DFA diagrams, flex/bison = classic Lex/Yacc version of the lexer
RUN apt-get update && apt-get install -y --no-install-recommends \
        graphviz flex bison build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install "torch>=2.7" --index-url ${TORCH_INDEX} \
    && pip install -r requirements.txt

RUN python -m nltk.downloader -d /usr/local/share/nltk_data punkt punkt_tab stopwords

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", \
     "--ServerApp.token=", "--ServerApp.password="]
