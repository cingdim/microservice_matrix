# Dockerfile.main
FROM python:3.12-slim

WORKDIR /app

# Copy source and requirements
COPY main.py .
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Wait for dependent services before starting main.py
CMD ["sh", "-c", "sleep 5 && python main.py"]
