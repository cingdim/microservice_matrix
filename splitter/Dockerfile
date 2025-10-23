# Dockerfile.splitter
FROM python:3.11-slim

WORKDIR /app

# Copy all necessary files into the container
COPY . /app

# Install dependencies for FastAPI + math operations
RUN pip install --no-cache-dir fastapi uvicorn numpy requests pydantic  python-multipart

# Expose the FastAPI service port
EXPOSE 8000

# Run the FastAPI app when container starts
CMD ["uvicorn", "splitter:app", "--host", "0.0.0.0", "--port", "8000"]
