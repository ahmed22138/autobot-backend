# =====================================================
# AutoBot Studio - Python FastAPI Backend
# =====================================================

FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies directly
RUN pip install --no-cache-dir \
    fastapi>=0.128.0 \
    uvicorn>=0.40.0 \
    supabase>=2.27.2 \
    openai>=1.0.0 \
    openai-agents>=0.7.0 \
    python-dotenv>=1.0.0

# Copy application code
COPY app/ ./app/
COPY chat.py .

# Expose port
EXPOSE 8000

# Run with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
