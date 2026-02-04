# ROYALEY - Production Dockerfile
# Enterprise-Grade Sports Prediction Platform
# Multi-stage build for optimal image size
#
# BUILD OPTIONS (use --build-arg):
#   INSTALL_AUTOGLUON=true  - Install AutoGluon (+2GB)
#   INSTALL_QUANTUM=true    - Install Quantum ML libraries (+500MB)
#   INSTALL_GPU=true        - Install GPU support (requires NVIDIA runtime)
#
# Examples:
#   docker compose up -d --build                    # Base only
#   docker compose up -d --build --build-arg INSTALL_AUTOGLUON=true
#   docker compose up -d --build --build-arg INSTALL_QUANTUM=true
#   docker compose up -d --build --build-arg INSTALL_AUTOGLUON=true --build-arg INSTALL_QUANTUM=true

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.11-slim as builder

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

# Copy all requirements files
COPY requirements.txt .
COPY requirements-autogluon.txt .
COPY requirements-quantum.txt .
COPY requirements-gpu.txt .

# Install base Python dependencies (always)
RUN pip wheel --no-cache-dir --wheel-dir=/build/wheels -r requirements.txt

# Install AutoGluon if requested (adds ~2GB to image)
RUN if [ "$INSTALL_AUTOGLUON" = "true" ]; then \
        echo "Installing AutoGluon dependencies..." && \
        pip wheel --no-cache-dir --wheel-dir=/build/wheels -r requirements-autogluon.txt; \
    fi

# Install Quantum ML if requested (adds ~500MB to image)
RUN if [ "$INSTALL_QUANTUM" = "true" ]; then \
        echo "Installing Quantum ML dependencies..." && \
        pip wheel --no-cache-dir --wheel-dir=/build/wheels -r requirements-quantum.txt; \
    fi

# Install GPU support if requested
RUN if [ "$INSTALL_GPU" = "true" ]; then \
        echo "Installing GPU dependencies..." && \
        pip wheel --no-cache-dir --wheel-dir=/build/wheels -r requirements-gpu.txt; \
    fi

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM python:3.11-slim as runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    APP_HOME=/app

WORKDIR $APP_HOME

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
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

# Create non-root user
RUN groupadd --gid 1000 appgroup \
    && useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Copy wheels from builder and install
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir /wheels/* \
    && rm -rf /wheels

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