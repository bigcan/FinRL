# Deployment Architecture & Strategies

**Feature**: SPY RL Trading System - Deployment Guide
**Phase**: Phase 11 (Docker & Containerization)
**Status**: Planning
**Last Updated**: 2025-10-30

---

## Overview

This document details the deployment architecture, Docker containerization strategy, and operational workflows for the SPY RL Trading System in production environments.

---

## Docker Architecture

### Multi-Stage Build Strategy

**Benefits**:
- **Smaller Images**: Final image only contains runtime dependencies (~400MB vs. ~1.2GB)
- **Faster Builds**: Layer caching accelerates rebuilds
- **Security**: Build tools not present in production image
- **Reproducibility**: Poetry lockfile ensures consistent dependencies

**Dockerfile Structure**:

```dockerfile
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Stage 1: Builder - Install dependencies
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FROM python:3.11-slim AS builder

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_VERSION=1.7.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s ${POETRY_HOME}/bin/poetry /usr/local/bin/poetry

WORKDIR /build

# Copy dependency files (leverage Docker layer caching)
COPY pyproject.toml poetry.lock ./

# Install dependencies (no root project, no dev dependencies)
RUN --mount=type=cache,target=$POETRY_CACHE_DIR \
    poetry install --no-root --only main --no-interaction --no-ansi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Stage 2: Runtime - Minimal production image
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FROM python:3.11-slim

# Create non-root user for security
RUN groupadd -r finrl && useradd -r -g finrl -m -s /bin/bash finrl

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /build/.venv /app/.venv

# Copy application code
COPY --chown=finrl:finrl finrl/ /app/finrl/

# Create directories with proper ownership
RUN mkdir -p /app/trained_models \
             /app/tensorboard_logs \
             /app/datasets \
             /app/results && \
    chown -R finrl:finrl /app

# Switch to non-root user
USER finrl

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import finrl; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "finrl.applications.spy_rl_trading.example_training"]
```

**Build Commands**:
```bash
# Build single-architecture image
docker build -t finrl-spy:latest .

# Build multi-architecture image (amd64 + arm64)
docker buildx create --use
docker buildx build \
    --platform linux/amd64,linux/arm64 \
    -t finrl-spy:latest \
    --push \
    .

# Build with build cache from registry
docker buildx build \
    --cache-from type=registry,ref=myregistry/finrl-spy:cache \
    --cache-to type=registry,ref=myregistry/finrl-spy:cache,mode=max \
    -t finrl-spy:latest \
    .
```

### Image Optimization

**Layer Caching Strategy**:
1. **Base OS Layer**: `FROM python:3.11-slim` (cached across builds)
2. **System Dependencies**: `apt-get install` (cached if unchanged)
3. **Poetry Installation**: Cached unless POETRY_VERSION changes
4. **Dependency Layer**: `poetry.lock` (cached if lockfile unchanged)
5. **Application Code**: `COPY finrl/` (changes most frequently)

**Size Optimization**:
- Use `slim` base image instead of full Python image
- Remove build tools in runtime stage
- Clean apt cache after installations
- Use `.dockerignore` to exclude unnecessary files

**.dockerignore**:
```
# Version control
.git/
.gitignore

# Python cache
__pycache__/
*.py[cod]
*$py.class
*.so

# Distribution
dist/
build/
*.egg-info/

# Testing
.pytest_cache/
.coverage
htmlcov/

# Development
.vscode/
.idea/
*.swp

# Data directories (too large, mounted as volumes)
trained_models/
tensorboard_logs/
datasets/
results/

# Documentation
docs/
*.md
LICENSE

# CI/CD
.github/
.gitlab-ci.yml
```

---

## Docker Compose Orchestration

### Service Architecture

**Services**:
1. **train**: Model training service
2. **backtest**: Backtesting service
3. **tensorboard**: TensorBoard visualization service
4. **paper-trading**: Paper trading service (optional)

**docker-compose.yml**:

```yaml
version: '3.8'

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Services
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
services:
  # Training Service
  train:
    build:
      context: .
      dockerfile: Dockerfile
      target: runtime
    image: finrl-spy:latest
    container_name: spy-training
    command: ["python", "-m", "finrl.applications.spy_rl_trading.example_training"]
    volumes:
      - trained_models:/app/trained_models
      - tensorboard_logs:/app/tensorboard_logs
      - datasets:/app/datasets
    environment:
      - SPY_LOG_LEVEL=INFO
      - SPY_LOG_JSON=true
      - SPY_RANDOM_SEED=42
      - SPY_TOTAL_TIMESTEPS=500000
      - SPY_LEARNING_RATE=0.0003
    networks:
      - finrl-network
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
    restart: "no"  # Don't restart on completion

  # Backtesting Service
  backtest:
    image: finrl-spy:latest
    container_name: spy-backtesting
    command: ["python", "-m", "finrl.applications.spy_rl_trading.example_backtesting"]
    volumes:
      - trained_models:/app/trained_models:ro  # Read-only
      - results:/app/results
      - datasets:/app/datasets:ro
    environment:
      - SPY_LOG_LEVEL=INFO
      - SPY_LOG_JSON=true
    networks:
      - finrl-network
    depends_on:
      train:
        condition: service_completed_successfully
    restart: "no"

  # TensorBoard Visualization Service
  tensorboard:
    image: tensorflow/tensorflow:latest
    container_name: spy-tensorboard
    command: ["tensorboard", "--logdir=/logs", "--host=0.0.0.0", "--port=6006"]
    ports:
      - "6006:6006"
    volumes:
      - tensorboard_logs:/logs:ro
    networks:
      - finrl-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6006"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Paper Trading Service (Optional)
  paper-trading:
    image: finrl-spy:latest
    container_name: spy-paper-trading
    command: ["python", "-m", "finrl.applications.spy_rl_trading.paper_trading"]
    volumes:
      - trained_models:/app/trained_models:ro
      - ./logs:/app/logs
    environment:
      - SPY_LOG_LEVEL=INFO
      - SPY_LOG_JSON=true
      - SPY_TRADING_MODE=paper
      - ALPACA_API_KEY=${ALPACA_API_KEY}
      - ALPACA_API_SECRET=${ALPACA_API_SECRET}
      - ALPACA_BASE_URL=https://paper-api.alpaca.markets
    networks:
      - finrl-network
    depends_on:
      - train
    restart: unless-stopped
    profiles:
      - paper-trading  # Only start with --profile paper-trading

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Networks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
networks:
  finrl-network:
    driver: bridge

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Volumes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
volumes:
  trained_models:
    driver: local
  tensorboard_logs:
    driver: local
  results:
    driver: local
  datasets:
    driver: local
```

### Operational Workflows

**Workflow 1: Full Training Pipeline**
```bash
# Build image
docker-compose build

# Run training
docker-compose up train

# Run backtest (after training completes)
docker-compose up backtest

# View TensorBoard
docker-compose up tensorboard
# Access at http://localhost:6006

# Cleanup
docker-compose down
```

**Workflow 2: Development with Live Code Updates**
```bash
# Override command to mount source code
docker-compose run --rm \
    -v $(pwd)/finrl:/app/finrl \
    train python -m finrl.applications.spy_rl_trading.example_training

# Or use docker-compose.override.yml for persistent dev config
cat > docker-compose.override.yml <<EOF
version: '3.8'
services:
  train:
    volumes:
      - ./finrl:/app/finrl
    environment:
      - SPY_LOG_LEVEL=DEBUG
EOF

docker-compose up train
```

**Workflow 3: Paper Trading**
```bash
# Set environment variables
export ALPACA_API_KEY=your_paper_key
export ALPACA_API_SECRET=your_paper_secret

# Start paper trading service
docker-compose --profile paper-trading up paper-trading
```

**Workflow 4: Parallel Training with Multiple Configs**
```bash
# docker-compose.parallel.yml
services:
  train-lr-high:
    extends:
      file: docker-compose.yml
      service: train
    container_name: spy-train-lr-high
    environment:
      - SPY_LEARNING_RATE=0.001
    volumes:
      - ./trained_models/lr_high:/app/trained_models

  train-lr-low:
    extends:
      file: docker-compose.yml
      service: train
    container_name: spy-train-lr-low
    environment:
      - SPY_LEARNING_RATE=0.0001
    volumes:
      - ./trained_models/lr_low:/app/trained_models

# Run parallel training
docker-compose -f docker-compose.parallel.yml up
```

---

## Production Deployment Strategies

### Strategy 1: Local Development

**Use Case**: Local development, testing, experimentation

**Setup**:
```bash
# Clone repository
git clone https://github.com/your-org/FinRL.git
cd FinRL

# Build and run
docker-compose up --build
```

**Pros**:
- Full control over code
- Fast iteration cycles
- No external dependencies

**Cons**:
- Manual management
- No high availability
- Limited scalability

---

### Strategy 2: Continuous Integration

**Use Case**: Automated testing, model validation, CI/CD pipelines

**GitHub Actions Workflow** (`.github/workflows/docker-build.yml`):
```yaml
name: Docker Build & Test

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Build Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: false
          tags: finrl-spy:test
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Run unit tests in container
        run: |
          docker run --rm finrl-spy:test \
            poetry run pytest unit_tests -m unit --cov=finrl

      - name: Run smoke test training
        run: |
          docker run --rm finrl-spy:test \
            python -m finrl.applications.spy_rl_trading.example_training \
            --timesteps 1024  # Quick smoke test
```

**Pros**:
- Automated testing
- Reproducible builds
- Quality gates before merge

**Cons**:
- CI compute costs
- Longer feedback loops

---

### Strategy 3: Cloud VM Deployment

**Use Case**: Long-running training jobs, scheduled backtesting, production paper trading

**Deployment Steps (AWS EC2 Example)**:

```bash
# 1. Launch EC2 instance (t3.xlarge or larger, Deep Learning AMI)
# 2. SSH into instance
ssh -i key.pem ec2-user@instance-ip

# 3. Install Docker
sudo yum update -y
sudo yum install -y docker
sudo service docker start
sudo usermod -a -G docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.23.0/docker-compose-$(uname -s)-$(uname -m)" \
    -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 4. Clone repository
git clone https://github.com/your-org/FinRL.git
cd FinRL

# 5. Set environment variables
cat > .env <<EOF
SPY_LOG_LEVEL=INFO
SPY_RANDOM_SEED=42
ALPACA_API_KEY=your_paper_key
ALPACA_API_SECRET=your_paper_secret
EOF

# 6. Run services
docker-compose up -d train tensorboard

# 7. Monitor logs
docker-compose logs -f train

# 8. Access TensorBoard (port forward)
# On local machine:
ssh -L 6006:localhost:6006 ec2-user@instance-ip
# Open http://localhost:6006
```

**Pros**:
- Dedicated compute resources
- Long-running jobs
- Persistent storage

**Cons**:
- Manual infrastructure management
- No auto-scaling
- Single point of failure

---

### Strategy 4: Kubernetes Orchestration

**Use Case**: High availability, auto-scaling, multi-model serving, production trading

**Kubernetes Manifests**:

**namespace.yaml**:
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: finrl-spy
```

**training-job.yaml**:
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: spy-training
  namespace: finrl-spy
spec:
  template:
    spec:
      containers:
      - name: training
        image: finrl-spy:latest
        command: ["python", "-m", "finrl.applications.spy_rl_trading.example_training"]
        env:
        - name: SPY_TOTAL_TIMESTEPS
          value: "500000"
        - name: SPY_LOG_LEVEL
          value: "INFO"
        volumeMounts:
        - name: trained-models
          mountPath: /app/trained_models
        - name: tensorboard-logs
          mountPath: /app/tensorboard_logs
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
      restartPolicy: OnFailure
      volumes:
      - name: trained-models
        persistentVolumeClaim:
          claimName: trained-models-pvc
      - name: tensorboard-logs
        persistentVolumeClaim:
          claimName: tensorboard-logs-pvc
```

**tensorboard-deployment.yaml**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tensorboard
  namespace: finrl-spy
spec:
  replicas: 1
  selector:
    matchLabels:
      app: tensorboard
  template:
    metadata:
      labels:
        app: tensorboard
    spec:
      containers:
      - name: tensorboard
        image: tensorflow/tensorflow:latest
        command: ["tensorboard", "--logdir=/logs", "--host=0.0.0.0"]
        ports:
        - containerPort: 6006
        volumeMounts:
        - name: tensorboard-logs
          mountPath: /logs
          readOnly: true
      volumes:
      - name: tensorboard-logs
        persistentVolumeClaim:
          claimName: tensorboard-logs-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: tensorboard
  namespace: finrl-spy
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 6006
  selector:
    app: tensorboard
```

**Deployment Commands**:
```bash
# Apply manifests
kubectl apply -f namespace.yaml
kubectl apply -f pvc.yaml
kubectl apply -f training-job.yaml
kubectl apply -f tensorboard-deployment.yaml

# Monitor training job
kubectl logs -f job/spy-training -n finrl-spy

# Access TensorBoard
kubectl get svc tensorboard -n finrl-spy
# Note the EXTERNAL-IP and access http://<external-ip>
```

**Pros**:
- High availability
- Auto-scaling
- Multi-model serving
- Declarative infrastructure

**Cons**:
- Complex setup
- Higher operational overhead
- Cost (managed Kubernetes)

---

## Resource Requirements

### Compute Resources

**Training (PPO Agent)**:
- **CPU**: 2-4 cores
- **Memory**: 4-8 GB
- **GPU**: Optional (2-3x speedup with CUDA)
- **Duration**: 30-60 minutes (500K timesteps)

**Backtesting**:
- **CPU**: 1-2 cores
- **Memory**: 2-4 GB
- **Duration**: 5-10 minutes

**TensorBoard**:
- **CPU**: 0.5-1 core
- **Memory**: 1-2 GB

**Paper Trading**:
- **CPU**: 0.5-1 core
- **Memory**: 1-2 GB
- **Network**: Stable internet connection

### Storage Requirements

- **Docker Image**: ~400 MB
- **Datasets**: ~50 MB (5 years daily data)
- **Trained Models**: ~10 MB per model
- **TensorBoard Logs**: ~100 MB per training run
- **Results**: ~5 MB per backtest

**Total**: ~1 GB for single training run with artifacts

---

## Monitoring & Observability

### Docker Health Checks

**Container Health**:
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import finrl; print('OK')" || exit 1
```

**Check Health**:
```bash
docker inspect --format='{{.State.Health.Status}}' spy-training
# Output: healthy | unhealthy | starting
```

### Log Aggregation

**Docker Logs**:
```bash
# View logs
docker-compose logs -f train

# Export logs
docker-compose logs train > training.log

# Filter JSON logs
docker-compose logs train | jq 'select(.level == "error")'
```

**Centralized Logging** (Optional):
```yaml
# docker-compose.yml logging section
services:
  train:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### Metrics Monitoring

**Prometheus Integration** (Optional):
```yaml
# docker-compose.monitoring.yml
services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    networks:
      - finrl-network

  grafana:
    image: grafana/grafana:latest
    volumes:
      - grafana_data:/var/lib/grafana
    ports:
      - "3000:3000"
    networks:
      - finrl-network
```

---

## Security Best Practices

### Container Security

**1. Non-Root User Execution**
```dockerfile
# Create non-root user
RUN useradd -r -m -s /bin/bash finrl
USER finrl
```

**2. Read-Only Filesystems**
```yaml
services:
  backtest:
    volumes:
      - trained_models:/app/trained_models:ro  # Read-only
    read_only: true
    tmpfs:
      - /tmp
```

**3. Secrets Management**
```bash
# Use Docker secrets
echo "my_secret_key" | docker secret create alpaca_api_key -

# Reference in compose
services:
  paper-trading:
    secrets:
      - alpaca_api_key
secrets:
  alpaca_api_key:
    external: true
```

**4. Network Isolation**
```yaml
networks:
  finrl-network:
    driver: bridge
    internal: true  # No external access
```

### Image Scanning

**Trivy Scan**:
```bash
# Install Trivy
curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin

# Scan image
trivy image finrl-spy:latest

# Fail on high/critical vulnerabilities
trivy image --severity HIGH,CRITICAL --exit-code 1 finrl-spy:latest
```

---

## Troubleshooting

### Common Issues

**Issue 1: Out of Memory**
```bash
# Symptom: Container killed (exit code 137)
docker-compose logs train | tail
# Output: Killed

# Solution: Increase memory limit
services:
  train:
    deploy:
      resources:
        limits:
          memory: 16G
```

**Issue 2: Volume Permission Errors**
```bash
# Symptom: Permission denied errors
# Solution: Ensure volume directories have correct ownership
sudo chown -R $(id -u):$(id -g) trained_models/ tensorboard_logs/
```

**Issue 3: GPU Not Detected**
```bash
# Symptom: Training slow, GPU not used
# Solution: Install nvidia-docker runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2

# Enable GPU in compose
services:
  train:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## Appendix

### Quick Reference

**Build Commands**:
```bash
docker build -t finrl-spy:latest .
docker-compose build
```

**Run Commands**:
```bash
docker-compose up train
docker-compose up backtest
docker-compose up tensorboard
docker-compose --profile paper-trading up paper-trading
```

**Management Commands**:
```bash
docker-compose ps
docker-compose logs -f train
docker-compose down
docker-compose down -v  # Remove volumes
```

**Debugging Commands**:
```bash
docker-compose exec train /bin/bash
docker-compose run --rm train python -c "import finrl; print(finrl.__version__)"
```

---

**Document Revision History**:
- 2025-10-30: Initial version covering Docker architecture and deployment strategies
