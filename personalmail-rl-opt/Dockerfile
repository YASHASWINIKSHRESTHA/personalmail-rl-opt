FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY . .

# Expose the FastAPI server port (OpenEnv standard)
EXPOSE 7863

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7863/health || exit 1

# Run the FastAPI server via uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7863"]
