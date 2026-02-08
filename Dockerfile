# ROYALEY - Production Dockerfile (Fixed)
# Enterprise-Grade Sports Prediction Platform
# Multi-stage build with proper dependency resolution
#
# BUILD OPTIONS (use --build-arg):
#   INSTALL_AUTOGLUON=true  - Install AutoGluon (+2GB)
#   INSTALL_QUANTUM=true    - Install Quantum ML libraries (+500MB)
#   INSTALL_GPU=true        - Install GPU support (PyTorch CUDA)
#
# Examples:
#   docker compose up -d --build                    # Base only (recommended)
#   docker compose up -d --build --build-arg INSTALL_AUTOGLUON=true
#   docker compose up -d --build --build-arg INSTALL_QUANTUM=true

# =============================================================================
# Stage 1: Builder - Create virtual environment with all dependencies
# =============================================================================
FROM python:3.11-slim AS builder

# Build arguments for optional dependencies
ARG INSTALL_AUTOGLUON=false
ARG INSTALL_QUANTUM=false
ARG INSTALL_GPU=false

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip for better dependency resolution
RUN pip install --upgrade pip setuptools wheel

# Copy all requirements files
COPY requirements.txt .
COPY requirements-autogluon.txt .
COPY requirements-quantum.txt .
COPY requirements-gpu.txt .

# =============================================================================
# Install dependencies in proper order with conflict resolution
# =============================================================================

# Step 1: Install base requirements (always)
RUN pip install --no-cache-dir -r requirements.txt

# Step 2: Install GPU dependencies if requested (before other ML libs to set torch version)
RUN if [ "$INSTALL_GPU" = "true" ]; then \
        echo "Installing GPU dependencies..." && \
        pip install --no-cache-dir \
            --extra-index-url https://download.pytorch.org/whl/cu121 \
            "torch==2.1.2+cu121" \
            "torchvision==0.16.2+cu121" \
            "torchaudio==2.1.2+cu121" \
            "cupy-cuda12x>=13.0.0" \
            "nvidia-ml-py>=12.535.133" \
            "pynvml>=11.5.0" \
            "gpustat>=1.1.1" \
            "onnxruntime-gpu>=1.16.0" \
        || echo "WARNING: Some GPU packages failed to install, continuing..."; \
        echo "Installing TensorFlow CUDA runtime libraries..." && \
        pip install --no-cache-dir \
            "nvidia-cublas-cu12" \
            "nvidia-cuda-cupti-cu12" \
            "nvidia-cuda-nvrtc-cu12" \
            "nvidia-cuda-runtime-cu12" \
            "nvidia-cudnn-cu12==8.9.*" \
            "nvidia-cufft-cu12" \
            "nvidia-curand-cu12" \
            "nvidia-cusolver-cu12" \
            "nvidia-cusparse-cu12" \
            "nvidia-nccl-cu12" \
            "nvidia-nvjitlink-cu12" \
        || echo "WARNING: Some TF CUDA packages failed, GPU may not work"; \
    fi

# Step 3: Install AutoGluon if requested (handles its own numpy/pandas/scipy versions)
RUN if [ "$INSTALL_AUTOGLUON" = "true" ]; then \
        echo "Installing AutoGluon dependencies..." && \
        pip install --no-cache-dir "autogluon.tabular>=1.0.0" \
        && echo "Installing Ray for parallel fold training..." && \
        pip install --no-cache-dir "ray>=2.43.0,<2.53.0" \
        || echo "WARNING: AutoGluon installation had issues, continuing..."; \
    fi

# Step 4: Install Quantum ML if requested (may update scipy)
RUN if [ "$INSTALL_QUANTUM" = "true" ]; then \
        echo "Installing Quantum ML dependencies..." && \
        pip install --no-cache-dir \
            "pennylane>=0.34.0" \
            "pennylane-lightning>=0.34.0" \
            "qiskit>=1.0.0" \
            "qiskit-aer>=0.13.0" \
            "qiskit-machine-learning>=0.7.0" \
            "qiskit-algorithms>=0.3.0" \
            "dwave-ocean-sdk>=6.5.0" \
            "dimod>=0.12.0" \
        || echo "WARNING: Some Quantum packages failed to install, continuing..."; \
    fi

# Step 5: CRITICAL - Ensure NumPy < 2.0 for pandas/pyarrow compatibility
# Some packages (especially quantum) upgrade numpy to 2.x which breaks pandas
RUN pip install --no-cache-dir "numpy>=1.26.4,<2.0.0" --force-reinstall

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM python:3.11-slim AS runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    APP_HOME=/app \
    PATH="/opt/venv/bin:$PATH"

# NVIDIA CUDA pip packages install shared libraries (.so) inside the venv.
# TensorFlow needs LD_LIBRARY_PATH to find libcudart, libcublas, libcudnn, etc.
# Without this, TF detects the GPU device but fails with "Could not find cuda drivers".
ENV LD_LIBRARY_PATH="/opt/venv/lib/python3.11/site-packages/nvidia/cublas/lib:\
/opt/venv/lib/python3.11/site-packages/nvidia/cuda_cupti/lib:\
/opt/venv/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:\
/opt/venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:\
/opt/venv/lib/python3.11/site-packages/nvidia/cudnn/lib:\
/opt/venv/lib/python3.11/site-packages/nvidia/cufft/lib:\
/opt/venv/lib/python3.11/site-packages/nvidia/curand/lib:\
/opt/venv/lib/python3.11/site-packages/nvidia/cusolver/lib:\
/opt/venv/lib/python3.11/site-packages/nvidia/cusparse/lib:\
/opt/venv/lib/python3.11/site-packages/nvidia/nccl/lib:\
/opt/venv/lib/python3.11/site-packages/nvidia/nvjitlink/lib\
${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

WORKDIR $APP_HOME

# Install runtime dependencies including Java for H2O
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    # Java JDK for H2O AutoML (default-jdk works across Debian versions)
    default-jdk-headless \
    # Chromium for Selenium (Action Network scraper)
    chromium \
    chromium-driver \
    # Required libraries for Chrome
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libatspi2.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2 \
    xdg-utils \
    # Additional libs for ML
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set Chrome environment variables for Selenium
ENV CHROME_BIN=/usr/bin/chromium \
    CHROMEDRIVER_PATH=/usr/bin/chromedriver

# Set Java environment for H2O (default-jdk installs to this path)
ENV JAVA_HOME=/usr/lib/jvm/default-java \
    PATH="/usr/lib/jvm/default-java/bin:$PATH"

# Create non-root user
RUN groupadd --gid 1000 appgroup \
    && useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY --chown=appuser:appgroup . .

# Create necessary directories (including kaggle config)
RUN mkdir -p /app/models /app/logs /app/data /app/ml_csv /home/appuser/.config/kaggle \
    && chown -R appuser:appgroup /app /home/appuser/.config

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Expose port
EXPOSE 8000

# Run application with Kaggle setup inline (creates kaggle.json from env var at startup)
CMD sh -c 'if [ -n "$KAGGLE_API_TOKEN" ]; then mkdir -p /home/appuser/.config/kaggle && echo "{\"username\":\"_\",\"key\":\"$KAGGLE_API_TOKEN\"}" > /home/appuser/.config/kaggle/kaggle.json && chmod 600 /home/appuser/.config/kaggle/kaggle.json; fi && uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4'