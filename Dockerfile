# ==============================================================================
# Dockerfile for the Pulmonary Nodule research service (not a medical device)
# Runs as a non-root user. Weights are NOT in git; mount/copy them before building.
# ==============================================================================

FROM python:3.10-slim AS runtime

# Prevent Python from writing .pyc and enable unbuffered output
ARG GIT_COMMIT_SHA=unknown
ENV GIT_COMMIT_SHA=${GIT_COMMIT_SHA} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PRODUCTION=1 \
    NODULE_DEMO=0

# Install system dependencies for OpenCV and DICOM parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root medical service user
RUN useradd -u 1001 -m -s /bin/bash appuser

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code and models
COPY model/ model/
# Fail the build instead of shipping an image that cannot serve predictions
RUN test -f model/best_model.pth && test -f model/calibrator.pth || \
    (echo 'ERROR: model/best_model.pth and model/calibrator.pth must exist at build time' && exit 1)
COPY preprocessing/ preprocessing/
COPY explainability/ explainability/
COPY expert_system/ expert_system/
COPY reports/ reports/
COPY service/ service/
COPY utils/ utils/
COPY scripts/ scripts/

# Create output directories with appropriate permissions
RUN mkdir -p /app/dist /app/storage && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Container health probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

